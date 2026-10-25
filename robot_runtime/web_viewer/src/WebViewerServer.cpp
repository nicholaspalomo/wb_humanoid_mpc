/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#include "web_viewer/WebViewerServer.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "httplib.h"  // NOLINT(build/include_subdir): cpp-httplib is a single header without a directory

#include "web_viewer/EncodedFrame.h"
#include "web_viewer/ThreadPlacement.h"
#include "web_viewer/ViewerFrameMailbox.h"
#include "web_viewer/ViewerInput.h"
#include "web_viewer/ViewerInputParsing.h"
#include "web_viewer/ViewerInputQueue.h"
#include "web_viewer/ViewerStatus.h"
#include "web_viewer/WebViewerPage.h"
#include "web_viewer/WebViewerServerOptions.h"

namespace robot::web_viewer {
namespace {

// LINT.IfChange(web_viewer_routes)
// The routes of README.md, "Routes", and what the page sends and reads.
constexpr char kPagePath[] = "/";
constexpr char kFramePath[] = "/frame.jpg";
constexpr char kStreamPath[] = "/stream.mjpg";
constexpr char kStatePath[] = "/state";
constexpr char kInputPath[] = "/input";
constexpr char kSettingsPath[] = "/settings";
// cpp-httplib matches routes as regular expressions: the same paths with their
// dots escaped.
constexpr char kFramePattern[] = R"(/frame\.jpg)";
constexpr char kStreamPattern[] = R"(/stream\.mjpg)";
constexpr char kAfterParameter[] = "after";
// The page's id on its frame requests, for /state's viewer count: 1 to
// kMaxViewerIdLength lower-case hex digits.
constexpr char kViewerParameter[] = "viewer";
constexpr size_t kMaxViewerIdLength = 32;
constexpr char kStreamBoundary[] = "mujocoframe";
// What follows each part's JPEG bytes.
constexpr absl::string_view kPartEnd = "\r\n";
// The header every POST carries: a cross-site page cannot send it without a
// CORS preflight, which is never granted.
constexpr char kViewerHeader[] = "X-MuJoCo-Viewer";
// How long GET /frame.jpg waits for a frame before 204 (with after=) or 503
// (without).
constexpr absl::Duration kFrameWait = absl::Seconds(2);
// How long a GET /stream.mjpg part waits for a new frame before cpp-httplib
// checks the client and asks again.
constexpr absl::Duration kStreamPartWait = absl::Seconds(1);
// LINT.ThenChange(//robot_runtime/web_viewer/page/index.html:web_viewer_routes, //robot_runtime/web_viewer/README.md:web_viewer_routes)

// The longest a waiting request sleeps between two checks of the server's stop
// flag: it bounds how long the destructor waits for a waiting request.
constexpr absl::Duration kWaitSlice = absl::Milliseconds(250);
// The bytes of one write of a frame. Linux reports a TCP socket writable once a
// third of its send buffer is free (the default buffer is 16 KiB), so a piece
// this small does not block in send(): a write to a client that stops reading
// waits only in cpp-httplib's writability check, for one write timeout, and not
// first in a send() of a whole frame for another. The server's stop is checked
// between two pieces.
constexpr size_t kWritePieceBytes = 4 * 1024;

// LINT.IfChange(web_viewer_limits)
// cpp-httplib's settings (README.md, "Limits").
constexpr int kKeepAliveTimeoutSeconds = 1;
constexpr size_t kKeepAliveMaxCount = 100;
constexpr size_t kPayloadMaxBytes = 1024;
constexpr int kReadTimeoutSeconds = 5;
constexpr int kWriteTimeoutSeconds = 2;
// Accepted connections that wait for a free request thread; beyond them a
// connection is closed at once.
constexpr size_t kMaxQueuedConnections = 16;
constexpr int kMaxClientsLimit = 64;
// The request pool: this many threads per frame client and a few more
// (README.md, "Limits").
constexpr int kRequestThreadsPerClient = 3;
constexpr int kExtraRequestThreads = 2;
// cpp-httplib's compile-time caps of a request, which README.md's Limits table
// quotes; an upgrade that changes one fails the build here. They are not
// defined here instead: the test client is another translation unit of
// httplib.h.
constexpr int kRequestUriMaxBytes = 8192;
constexpr int kHeaderLineMaxBytes = 8192;
constexpr int kHeaderMaxCount = 100;
static_assert(CPPHTTPLIB_REQUEST_URI_MAX_LENGTH == kRequestUriMaxBytes,
              "cpp-httplib's URI cap changed: update README.md, Limits");
static_assert(
    CPPHTTPLIB_HEADER_MAX_LENGTH == kHeaderLineMaxBytes,
    "cpp-httplib's header line cap changed: update README.md, Limits");
static_assert(
    CPPHTTPLIB_HEADER_MAX_COUNT == kHeaderMaxCount,
    "cpp-httplib's header count cap changed: update README.md, Limits");
// LINT.ThenChange(//robot_runtime/web_viewer/README.md:web_viewer_limits)
// The loopback network of IPv4, 127.0.0.0/8: its first octet, the top eight
// bits of an address.
constexpr int kIpv4FirstOctetShift = 24;
constexpr int kIpv4LoopbackFirstOctet = 127;
// [s] How often a refused request is logged at most.
constexpr int kRefusedRequestLogPeriodSeconds = 10;
// An idle renderer's frame older than this many frame periods is stale.
constexpr int kStaleFramePeriods = 2;

constexpr int kNoContent = 204;
constexpr int kBadRequest = 400;
constexpr int kForbidden = 403;
constexpr int kNotFound = 404;
constexpr int kMethodNotAllowed = 405;
constexpr int kServiceUnavailable = 503;

// The routes, for the 404 and 405 answers of the checks before routing.
enum class Route {
  kPage,
  kFrame,
  kStream,
  kState,
  kInput,
  kSettings,
};

std::optional<Route> routeOf(absl::string_view path) {
  if (path == kPagePath) return Route::kPage;
  if (path == kFramePath) return Route::kFrame;
  if (path == kStreamPath) return Route::kStream;
  if (path == kStatePath) return Route::kState;
  if (path == kInputPath) return Route::kInput;
  if (path == kSettingsPath) return Route::kSettings;
  return std::nullopt;
}

bool isPostRoute(Route route) {
  return route == Route::kInput || route == Route::kSettings;
}

// A bind address in its canonical text form (inet_ntop), and what kind of
// address it is.
struct BindAddress {
  std::string canonical = "";
  bool isIpv6 = false;
  bool isWildcard = false;
  bool isLoopback = false;
};

// The text form of a binary address of `family` (inet_ntop), empty if it has
// none.
std::string addressText(int family, const void* absl_nonnull binary) {
  std::array<char, INET6_ADDRSTRLEN> buffer = {};
  const char* absl_nullable text =
      inet_ntop(family, binary, buffer.data(), buffer.size());
  return text == nullptr ? std::string() : std::string(text);
}

absl::StatusOr<BindAddress> parseBindAddress(absl::string_view address) {
  const std::string text(address);
  in_addr ipv4 = {};
  if (inet_pton(AF_INET, text.c_str(), &ipv4) == 1) {
    const uint32_t hostOrder = ntohl(ipv4.s_addr);
    return BindAddress{.canonical = addressText(AF_INET, &ipv4),
                       .isIpv6 = false,
                       .isWildcard = hostOrder == INADDR_ANY,
                       .isLoopback = (hostOrder >> kIpv4FirstOctetShift) ==
                                     kIpv4LoopbackFirstOctet};
  }
  in6_addr ipv6 = {};
  if (inet_pton(AF_INET6, text.c_str(), &ipv6) == 1) {
    return BindAddress{.canonical = addressText(AF_INET6, &ipv6),
                       .isIpv6 = true,
                       .isWildcard = IN6_IS_ADDR_UNSPECIFIED(&ipv6) != 0,
                       .isLoopback = IN6_IS_ADDR_LOOPBACK(&ipv6) != 0};
  }
  return absl::InvalidArgumentError(
      absl::StrCat("the web viewer binds a numeric IPv4 or IPv6 address "
                   "(127.0.0.1, ::1, 0.0.0.0, ::), not '",
                   address, "'"));
}

// `name` in inet_ntop's form when it is an IPv6 address ("2001:0DB8:0:0::1" is
// "2001:db8::1", "::ffff:c0a8:105" is
// "::ffff:192.168.1.5"), else unchanged: a browser and an allowed-host entry
// may spell one address differently.
std::string canonicalHostname(std::string name) {
  in6_addr ipv6 = {};
  if (inet_pton(AF_INET6, name.c_str(), &ipv6) == 1)
    return addressText(AF_INET6, &ipv6);
  return name;
}

// The host name of a Host header without its port, lower case, an IPv6 address
// in inet_ntop's form: "LocalHost:6090" is "localhost", "[0:0::1]:6090" is
// "::1". Nullopt for a header that is empty or malformed (an IPv6 address
// without brackets).
std::optional<std::string> hostnameOf(absl::string_view host) {
  if (host.empty()) return std::nullopt;
  if (host.front() == '[') {
    const size_t close = host.find(']');
    if (close == absl::string_view::npos || close == 1) return std::nullopt;
    const absl::string_view rest = host.substr(close + 1);
    if (!rest.empty() && rest.front() != ':') return std::nullopt;
    return canonicalHostname(absl::AsciiStrToLower(host.substr(1, close - 1)));
  }
  const size_t colon = host.find(':');
  if (colon == 0 || (colon != absl::string_view::npos &&
                     host.find(':', colon + 1) != absl::string_view::npos))
    return std::nullopt;
  return absl::AsciiStrToLower(host.substr(0, colon));
}

// The refusal of an allowed-host entry that is no host name or address.
absl::Status notAnAllowedHost(absl::string_view entry) {
  return absl::InvalidArgumentError(
      absl::StrCat("an allowed host (--web_viewer_allowed_hosts) is a host "
                   "name or address without a port, not '",
                   entry, "'"));
}

// The host name an allowed-host entry names, lower case, in the form
// hostnameOf() reads from a Host header: "Robot.LAN" is "robot.lan", and
// "[FE80:0::1]" and "fe80::1" are "fe80::1". InvalidArgument for an entry with
// a port
// ("robot.lan:6090", "[fe80::1]:6090"), which no request's host name equals,
// and for one that is empty, holds a '/' or a space, or brackets anything but
// an IPv6 address.
absl::StatusOr<std::string> allowedHostnameOf(absl::string_view entry) {
  if (entry.empty() || absl::StrContains(entry, '/') ||
      absl::StrContains(entry, ' '))
    return notAnAllowedHost(entry);
  const bool bracketed = entry.front() == '[';
  if (!bracketed && std::count(entry.begin(), entry.end(), /*value=*/':') > 1) {
    // An IPv6 address without the brackets a Host header puts around it.
    in6_addr ipv6 = {};
    if (inet_pton(AF_INET6, std::string(entry).c_str(), &ipv6) != 1)
      return notAnAllowedHost(entry);
    return addressText(AF_INET6, &ipv6);
  }
  const std::optional<std::string> hostname = hostnameOf(entry);
  if (!hostname.has_value()) return notAnAllowedHost(entry);
  // hostnameOf() drops what follows the name: "name:port", "[v6]:port", or a
  // bare ':'.
  const bool hasPort =
      bracketed ? entry.back() != ']' : absl::StrContains(entry, ':');
  if (hasPort) {
    return absl::InvalidArgumentError(
        absl::StrCat("an allowed host (--web_viewer_allowed_hosts) is a name "
                     "without a port, since a request is "
                     "matched by its host name alone: give '",
                     *hostname, "', not '", entry, "'"));
  }
  in6_addr ipv6 = {};
  if (bracketed && inet_pton(AF_INET6, hostname->c_str(), &ipv6) != 1)
    return notAnAllowedHost(entry);
  return *hostname;
}

absl::Status checkOptions(const WebViewerServerOptions& options,
                          const ViewerFrameMailbox& mailbox) {
  if (options.port < 0 || options.port > WebViewerServerOptions::kMaxPort) {
    return absl::InvalidArgumentError(
        absl::StrCat("the web viewer port is 0 (any free port) to ",
                     WebViewerServerOptions::kMaxPort, ", not ", options.port));
  }
  if (options.maxClients < 1 || options.maxClients > kMaxClientsLimit) {
    return absl::InvalidArgumentError(
        absl::StrCat("the web viewer serves 1 to ", kMaxClientsLimit,
                     " frame clients at a time, not ", options.maxClients));
  }
  if (options.maxClients != mailbox.maxClients()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "the server's maxClients (", options.maxClients,
        ") differs from its mailbox's (", mailbox.maxClients(), ")"));
  }
  for (const std::string& host : options.allowedHosts) {
    const absl::StatusOr<std::string> hostname = allowedHostnameOf(host);
    if (!hostname.ok()) return hostname.status();
  }
  return absl::OkStatus();
}

void answer(httplib::Response& response,
            int status,
            absl::string_view message) {
  response.status = status;
  response.set_content(absl::StrCat(message, "\n"),
                       "text/plain; charset=utf-8");
}

// The query parameters of a request by their decoded names, read from the raw
// request target: cpp-httplib's own `params` drop an exact repeat
// ("key=a&key=a") and add the parameters of a form body. InvalidArgument when
// one is repeated or the request has a body.
absl::StatusOr<absl::flat_hash_map<std::string, std::string>> singleParameters(
    const httplib::Request& request) {
  if (!request.body.empty()) {
    return absl::InvalidArgumentError(
        "the routes take query parameters only, not a request body");
  }
  absl::flat_hash_map<std::string, std::string> parameters;
  const size_t question = request.target.find('?');
  if (question == std::string::npos) return parameters;
  for (const absl::string_view segment : absl::StrSplit(
           absl::string_view(request.target).substr(question + 1), '&')) {
    const std::pair<absl::string_view, absl::string_view> keyValue =
        absl::StrSplit(segment, absl::MaxSplits('=', 1));
    if (keyValue.first.empty()) continue;  // as cpp-httplib skips "&&" and "=x"
    std::string name =
        httplib::decode_query_component(std::string(keyValue.first));
    if (!parameters
             .try_emplace(name, httplib::decode_query_component(
                                    std::string(keyValue.second)))
             .second) {
      return absl::InvalidArgumentError(
          absl::StrCat("parameter '", name, "' is given more than once"));
    }
  }
  return parameters;
}

// Whether `id` is a page id of the frame route: 1 to kMaxViewerIdLength
// lower-case hex digits.
bool isViewerId(absl::string_view id) {
  return !id.empty() && id.size() <= kMaxViewerIdLength &&
         std::all_of(id.begin(), id.end(), [](char character) {
           return absl::ascii_isdigit(character) ||
                  (character >= 'a' && character <= 'f');
         });
}

// Whether `origin` is the page under `host` itself: over plain HTTP, or behind
// an HTTPS-terminating forward that keeps the Host. Any other scheme, name or
// port is another page's.
bool isOwnOrigin(absl::string_view origin, absl::string_view host) {
  return absl::EqualsIgnoreCase(origin, absl::StrCat("http://", host)) ||
         absl::EqualsIgnoreCase(origin, absl::StrCat("https://", host));
}

// The milliseconds since `frame` was published, on the monotonic clock.
int64_t frameAgeMilliseconds(const EncodedFrame& frame) {
  return absl::ToInt64Milliseconds(absl::FromChrono(
      std::chrono::steady_clock::now() - frame.publishSteadyTime));
}

std::string jsonNumber(double value) {
  return std::isfinite(value) ? absl::StrFormat("%.9g", value) : "null";
}

}  // namespace

/**
 * The server's state and handlers. Lives on the heap at a fixed address, which
 * the handlers and the listen thread capture; WebViewerServer owns it. The
 * handlers run on the request threads and touch only the mailbox, the input
 * queue and the immutable configuration.
 */
class WebViewerServer::Core {
 public:
  static absl::StatusOr<std::unique_ptr<Core>> Create(
      WebViewerServerOptions options,
      BindAddress address,
      ViewerFrameMailbox* absl_nonnull mailbox,
      ViewerInputQueue* absl_nonnull inputs);

  /** Stops accepting connections, ends the open requests and joins the listen
   * thread and the request pool. */
  ~Core();
  Core(const Core&) = delete;
  Core& operator=(const Core&) = delete;

  int port() const { return port_; }
  std::string url() const;

 private:
  Core(WebViewerServerOptions options,
       BindAddress address,
       ViewerFrameMailbox* absl_nonnull mailbox,
       ViewerInputQueue* absl_nonnull inputs);

  void configureServer();
  void installRoutes();
  // Binds the socket and then serves until stop(), on the listen thread.
  void runListenThread();
  absl::Status bind();

  bool stopped() const { return stopping_.load() || mailbox_->closed(); }
  httplib::Server::HandlerResponse checkRequest(
      const httplib::Request& request, httplib::Response& response) const;
  bool hostAllowed(absl::string_view hostname) const {
    return allowedHostnames_.contains(hostname);
  }
  static void addSecurityHeaders(httplib::Response& response);

  void servePage(httplib::Response& response) const;
  void serveFrame(const httplib::Request& request,
                  httplib::Response& response) const;
  void serveStream(httplib::Response& response) const;
  void serveState(httplib::Response& response) const;
  void serveInput(const httplib::Request& request,
                  httplib::Response& response) const;
  void serveSettings(const httplib::Request& request,
                     httplib::Response& response) const;

  // The newest frame with a sequence above `afterSequence`, waiting up to
  // `timeout` in slices of kWaitSlice; nullopt on the timeout and once the
  // server stops or the mailbox closes.
  std::optional<EncodedFrame> waitForFrame(int64_t afterSequence,
                                           absl::Duration timeout) const;
  // Whether `frame` was published before the frame request that last found the
  // renderer without demand (ViewerFrameMailbox::demandResumedAt()) and is
  // older than two periods of the target rate (at most the demand lease): it
  // was drawn before the renderer went idle. A frame of a busy renderer is
  // never stale, however old.
  bool isStale(const EncodedFrame& frame) const;
  // GET /frame.jpg without after=: the newest frame if it is fresh, else the
  // next one.
  std::optional<EncodedFrame> freshFrame() const;
  // GET /frame.jpg?after=: the newest frame above `afterSequence`, and the next
  // one instead when that is stale.
  std::optional<EncodedFrame> frameAfter(int64_t afterSequence) const;
  // Writes the next part of a GET /stream.mjpg; false ends the stream.
  bool writeStreamPart(int64_t& lastSequence, httplib::DataSink& sink) const;
  // Writes `bytes` to `sink` in pieces of kWritePieceBytes; false once a write
  // fails or the server stops.
  bool writeInPieces(absl::string_view bytes, httplib::DataSink& sink) const;
  void answerUnavailable(httplib::Response& response) const;

  const WebViewerServerOptions options_;
  const BindAddress address_;
  ViewerFrameMailbox* absl_nonnull const mailbox_;
  ViewerInputQueue* absl_nonnull const inputs_;
  // The host names a request's Host header may carry: the loopback names, a
  // non-wildcard bind address and options_.allowedHosts, lower case.
  absl::flat_hash_set<std::string> allowedHostnames_;
  httplib::Server server_;
  std::atomic<bool> stopping_ = false;
  // The open GET /stream.mjpg responses, each a viewer of /state.
  mutable std::atomic<int> openStreams_ = 0;
  // Written by the listen thread before bound_ is notified, read after: the
  // bound port (0 until then) and the result of the bind.
  int port_ = 0;
  absl::Status bindStatus_;
  absl::Notification bound_;
  // Written by the listen thread before listening_ is notified, read after: OK
  // once the request pool exists and the accept loop starts, the reason
  // otherwise. Notified exactly once after a successful bind.
  absl::Status listenStatus_;
  absl::Notification listening_;
  std::thread listenThread_;
};

absl::StatusOr<std::unique_ptr<WebViewerServer::Core>>
WebViewerServer::Core::Create(WebViewerServerOptions options,
                              BindAddress address,
                              ViewerFrameMailbox* absl_nonnull mailbox,
                              ViewerInputQueue* absl_nonnull inputs) {
  std::unique_ptr<Core> core = absl::WrapUnique(
      new Core(std::move(options), std::move(address), mailbox, inputs));
  try {  // NOLINT(exceptions): cpp-httplib throws (std::regex in its route matchers, std::system_error from std::thread)
    core->installRoutes();
    core->listenThread_ =
        std::thread([core = core.get()] { core->runListenThread(); });
    // NOLINTNEXTLINE(exceptions): converts cpp-httplib's and std::thread's exceptions at once
  } catch (const std::exception& error) {
    return absl::InternalError(
        absl::StrCat("the web viewer server could not start: ", error.what()));
  }
  core->bound_.WaitForNotification();
  if (!core->bindStatus_.ok()) return core->bindStatus_;
  // Not server_.wait_until_ready(): cpp-httplib reports itself running before
  // it builds its request pool, which can still fail.
  core->listening_.WaitForNotification();
  if (!core->listenStatus_.ok()) return core->listenStatus_;
  return core;
}

WebViewerServer::Core::Core(WebViewerServerOptions options,
                            BindAddress address,
                            ViewerFrameMailbox* absl_nonnull mailbox,
                            ViewerInputQueue* absl_nonnull inputs)
    : options_(std::move(options)),
      address_(std::move(address)),
      mailbox_(mailbox),
      inputs_(inputs) {
  allowedHostnames_ = {"localhost", "127.0.0.1", "::1"};
  if (!address_.isWildcard) allowedHostnames_.insert(address_.canonical);
  for (const std::string& host : options_.allowedHosts) {
    // Create() has refused every entry that does not parse (checkOptions()).
    absl::StatusOr<std::string> hostname = allowedHostnameOf(host);
    if (hostname.ok()) allowedHostnames_.insert(*std::move(hostname));
  }
  configureServer();
}

WebViewerServer::Core::~Core() {
  stopping_.store(true);
  if (!listenThread_.joinable()) return;
  bound_.WaitForNotification();
  // stop() does nothing before the listen thread runs its accept loop, which a
  // failed bind never reaches.
  if (bindStatus_.ok()) server_.wait_until_ready();
  server_.stop();
  listenThread_.join();
}

void WebViewerServer::Core::configureServer() {
  // SO_REUSEADDR only: cpp-httplib's default also sets SO_REUSEPORT, which
  // would let a second process bind the same port and take half of the
  // connections.
  server_.set_socket_options([](int socket) {
    httplib::set_socket_opt(socket, SOL_SOCKET, SO_REUSEADDR, /*optval=*/1);
  });
  const size_t poolSize = static_cast<size_t>(
      kRequestThreadsPerClient * options_.maxClients + kExtraRequestThreads);
  // cpp-httplib takes ownership of the pool it gets as a raw pointer. It builds
  // the pool on the listen thread just before its accept loop, on a socket that
  // already listens: once the pool exists, the server accepts.
  server_.new_task_queue = [this, poolSize] {
    std::unique_ptr<httplib::ThreadPool> pool =
        std::make_unique<httplib::ThreadPool>(poolSize, poolSize,
                                              kMaxQueuedConnections);
    listening_.Notify();
    return pool.release();
  };
  server_.set_tcp_nodelay(/*on=*/true);
  server_.set_keep_alive_timeout(kKeepAliveTimeoutSeconds);
  server_.set_keep_alive_max_count(kKeepAliveMaxCount);
  server_.set_payload_max_length(kPayloadMaxBytes);
  server_.set_read_timeout(kReadTimeoutSeconds, /*usec=*/0);
  server_.set_write_timeout(kWriteTimeoutSeconds, /*usec=*/0);
  server_.set_pre_routing_handler(
      [this](const httplib::Request& request, httplib::Response& response) {
        return checkRequest(request, response);
      });
  server_.set_post_routing_handler(
      [](const httplib::Request& /*request*/, httplib::Response& response) {
        addSecurityHeaders(response);
      });
}

void WebViewerServer::Core::installRoutes() {
  server_.Get(kPagePath,
              [this](const httplib::Request& /*request*/,
                     httplib::Response& response) { servePage(response); });
  server_.Get(kFramePattern, [this](const httplib::Request& request,
                                    httplib::Response& response) {
    serveFrame(request, response);
  });
  server_.Get(kStreamPattern,
              [this](const httplib::Request& /*request*/,
                     httplib::Response& response) { serveStream(response); });
  server_.Get(kStatePath,
              [this](const httplib::Request& /*request*/,
                     httplib::Response& response) { serveState(response); });
  server_.Post(kInputPath, [this](const httplib::Request& request,
                                  httplib::Response& response) {
    serveInput(request, response);
  });
  server_.Post(kSettingsPath, [this](const httplib::Request& request,
                                     httplib::Response& response) {
    serveSettings(request, response);
  });
}

void WebViewerServer::Core::runListenThread() {
  const absl::Status placed =
      placeCurrentThread(options_.cores, options_.threadPriority);
  if (!placed.ok())
    LOG(WARNING) << "[WebViewerServer] the server's threads run unplaced: "
                 << placed;
  bindStatus_ = bind();
  bound_.Notify();
  if (!bindStatus_.ok()) return;
  try {  // NOLINT(exceptions): cpp-httplib's thread pool rethrows the std::system_error of a thread that cannot start
    if (!server_.listen_after_bind() && !stopping_.load())
      LOG(ERROR) << "[WebViewerServer] stopped accepting connections at "
                 << url();
    // NOLINTNEXTLINE(exceptions): converted to a log line, and to Create()'s Internal
  } catch (const std::exception& error) {
    LOG(ERROR) << "[WebViewerServer] could not serve " << url() << ": "
               << error.what();
    server_.decommission();
    if (!listening_.HasBeenNotified()) {
      listenStatus_ = absl::InternalError(absl::StrCat(
          "the web viewer server bound ", url(),
          " but could not start its request threads: ", error.what()));
      listening_.Notify();
    }
    return;
  }
  // listen_after_bind() returned without building the pool: stopped before it
  // started.
  if (!listening_.HasBeenNotified()) {
    listenStatus_ = absl::InternalError(
        absl::StrCat("the web viewer server bound ", url(),
                     " but could not listen (see the log)"));
    listening_.Notify();
  }
}

absl::Status WebViewerServer::Core::bind() {
  constexpr int kNumericAddress = AI_NUMERICHOST | AI_PASSIVE;
  int boundPort = options_.port;
  bool bound = false;
  if (options_.port == 0) {
    boundPort = server_.bind_to_any_port(address_.canonical, kNumericAddress);
    bound = boundPort > 0;
  } else {
    bound = server_.bind_to_port(address_.canonical, options_.port,
                                 kNumericAddress);
  }
  const int error = errno;
  if (!bound) {
    const absl::Status cause = absl::ErrnoToStatus(error, "bind");
    return absl::UnavailableError(absl::StrCat(
        "the web viewer could not bind ", address_.canonical, ":",
        options_.port, " (", cause.message(),
        "): another process, such as a second robot or simulation, may hold "
        "the port; stop it "
        "(make kill-sims) or choose another port (--web_viewer_port)"));
  }
  port_ = boundPort;
  return absl::OkStatus();
}

std::string WebViewerServer::Core::url() const {
  if (address_.isWildcard) return absl::StrCat("http://localhost:", port_, "/");
  if (address_.isIpv6)
    return absl::StrCat("http://[", address_.canonical, "]:", port_, "/");
  return absl::StrCat("http://", address_.canonical, ":", port_, "/");
}

httplib::Server::HandlerResponse WebViewerServer::Core::checkRequest(
    const httplib::Request& request, httplib::Response& response) const {
  const std::string host = request.get_header_value("Host");
  const std::optional<std::string> hostname = hostnameOf(host);
  if (!hostname.has_value() || !hostAllowed(*hostname)) {
    LOG_EVERY_N_SEC(WARNING, kRefusedRequestLogPeriodSeconds)
        << "[WebViewerServer] refused a request for Host '" << host
        << "' (allowed: localhost, 127.0.0.1, ::1"
        << (address_.isWildcard ? "" : absl::StrCat(", ", address_.canonical))
        << " and the allowed hosts)";
    answer(response, kForbidden,
           absl::StrCat(
               "Host '", host,
               "' is not allowed: open the viewer at http://localhost:", port_,
               "/ (an SSH or VS Code tunnel to it works), or add the name to "
               "the allowed hosts (--web_viewer_allowed_hosts)"));
    return httplib::Server::HandlerResponse::Handled;
  }
  const std::string fetchSite =
      absl::AsciiStrToLower(request.get_header_value("Sec-Fetch-Site"));
  if (fetchSite == "cross-site" || fetchSite == "same-site") {
    answer(response, kForbidden,
           absl::StrCat("the viewer serves its own page only (Sec-Fetch-Site: ",
                        fetchSite, ")"));
    return httplib::Server::HandlerResponse::Handled;
  }
  const std::optional<Route> route = routeOf(request.path);
  if (!route.has_value()) {
    answer(response, kNotFound, "not found");
    return httplib::Server::HandlerResponse::Handled;
  }
  const bool post = isPostRoute(*route);
  const bool methodAllowed =
      post ? request.method == "POST"
           : (request.method == "GET" || request.method == "HEAD");
  if (!methodAllowed) {
    response.set_header("Allow", post ? "POST" : "GET, HEAD");
    answer(response, kMethodNotAllowed,
           absl::StrCat(request.path, " takes ", post ? "POST" : "GET"));
    return httplib::Server::HandlerResponse::Handled;
  }
  if (post) {
    if (request.get_header_value(kViewerHeader) != "1") {
      answer(response, kForbidden,
             absl::StrCat("a POST needs the header ", kViewerHeader, ": 1"));
      return httplib::Server::HandlerResponse::Handled;
    }
    if (request.has_header("Origin") &&
        !isOwnOrigin(request.get_header_value("Origin"), host)) {
      answer(response, kForbidden,
             absl::StrCat("a POST from origin '",
                          request.get_header_value("Origin"), "' is refused"));
      return httplib::Server::HandlerResponse::Handled;
    }
  }
  return httplib::Server::HandlerResponse::Unhandled;
}

void WebViewerServer::Core::addSecurityHeaders(httplib::Response& response) {
  response.set_header("Cross-Origin-Resource-Policy", "same-origin");
  response.set_header("X-Frame-Options", "DENY");
  response.set_header("Content-Security-Policy", "frame-ancestors 'none'");
  response.set_header("X-Content-Type-Options", "nosniff");
  if (!response.has_header("Cache-Control"))
    response.set_header("Cache-Control", "no-store");
}

void WebViewerServer::Core::servePage(httplib::Response& response) const {
  response.set_header("Cache-Control", "no-store");
  response.set_content(std::string(webViewerPage()),
                       "text/html; charset=utf-8");
}

void WebViewerServer::Core::answerUnavailable(
    httplib::Response& response) const {
  answer(response, kServiceUnavailable, "the viewer has stopped");
}

std::optional<EncodedFrame> WebViewerServer::Core::waitForFrame(
    int64_t afterSequence, absl::Duration timeout) const {
  const absl::Time deadline = absl::Now() + timeout;
  while (!stopped()) {
    const absl::Duration remaining = deadline - absl::Now();
    if (remaining <= absl::ZeroDuration()) return std::nullopt;
    std::optional<EncodedFrame> frame =
        mailbox_->copyNewer(afterSequence, std::min(remaining, kWaitSlice));
    if (frame.has_value()) return frame;
  }
  return std::nullopt;
}

bool WebViewerServer::Core::isStale(const EncodedFrame& frame) const {
  const std::optional<std::chrono::steady_clock::time_point> resumed =
      mailbox_->demandResumedAt();
  if (!resumed.has_value() || frame.publishSteadyTime >= *resumed) return false;
  // A frame the renderer finished just before it went idle is still current;
  // one older than two frame periods (at most the demand lease, after which an
  // idle renderer has drawn nothing) is not.
  const absl::Duration framePeriod =
      absl::Seconds(1) / std::max(mailbox_->status().render.targetFps, 1);
  return absl::FromChrono(std::chrono::steady_clock::now() -
                          frame.publishSteadyTime) >=
         std::min(kStaleFramePeriods * framePeriod,
                  ViewerFrameMailbox::kDemandLease);
}

std::optional<EncodedFrame> WebViewerServer::Core::freshFrame() const {
  std::optional<EncodedFrame> latest =
      mailbox_->copyNewer(/*afterSequence=*/0, absl::ZeroDuration());
  if (latest.has_value() && !isStale(*latest)) return latest;
  return waitForFrame(latest.has_value() ? latest->sequence : 0, kFrameWait);
}

std::optional<EncodedFrame> WebViewerServer::Core::frameAfter(
    int64_t afterSequence) const {
  // A sequence beyond the newest comes from a page of an earlier viewer (the
  // simulation restarted): it gets the newest frame.
  const std::optional<EncodedFrame> latest = mailbox_->latestFrameMetadata();
  const int64_t newest = latest.has_value() ? latest->sequence : 0;
  std::optional<EncodedFrame> frame =
      waitForFrame(afterSequence > newest ? 0 : afterSequence, kFrameWait);
  if (!frame.has_value() || !isStale(*frame)) return frame;
  // The renderer was idle, so the page came back to a frame of then (a tab
  // shown again, a new tab); this request has woken the renderer, and its next
  // frame follows within one render. If none does, the stale frame is still
  // newer than what the page shows.
  std::optional<EncodedFrame> fresh = waitForFrame(frame->sequence, kFrameWait);
  if (fresh.has_value()) return fresh;
  return frame;
}

void WebViewerServer::Core::serveFrame(const httplib::Request& request,
                                       httplib::Response& response) const {
  std::optional<int64_t> after;
  if (request.has_param(kAfterParameter)) {
    int64_t value = 0;
    if (!absl::SimpleAtoi(request.get_param_value(kAfterParameter), &value) ||
        value < 0) {
      answer(response, kBadRequest,
             "parameter 'after' is a frame sequence, an integer >= 0");
      return;
    }
    after = value;
  }
  std::optional<std::string> viewerId;
  if (request.has_param(kViewerParameter)) {
    viewerId = request.get_param_value(kViewerParameter);
    if (!isViewerId(*viewerId)) {
      answer(response, kBadRequest,
             absl::StrCat("parameter 'viewer' is 1 to ", kMaxViewerIdLength,
                          " lower-case hex digits"));
      return;
    }
  }
  if (stopped()) {
    answerUnavailable(response);
    return;
  }
  if (!mailbox_->tryAcquireClient()) {
    answer(response, kServiceUnavailable,
           absl::StrCat("too many viewers (", mailbox_->maxClients(),
                        "): close another viewer tab or stream"));
    return;
  }
  // A page counts as a viewer from its request on and for the lease after its
  // answer, also after a 2 s long poll.
  absl::Cleanup release = [this, &viewerId] {
    if (viewerId.has_value()) mailbox_->noteViewer(*viewerId);
    mailbox_->releaseClient();
  };
  mailbox_->noteFrameRequest();
  if (viewerId.has_value()) mailbox_->noteViewer(*viewerId);
  std::optional<EncodedFrame> frame =
      after.has_value() ? frameAfter(*after) : freshFrame();
  if (!frame.has_value()) {
    if (stopped()) {
      answerUnavailable(response);
    } else if (after.has_value()) {
      response.status = kNoContent;
    } else {
      answer(response, kServiceUnavailable,
             absl::StrCat("no frame within ", absl::FormatDuration(kFrameWait),
                          ": is the renderer running?"));
    }
    return;
  }
  response.set_header("X-Frame-Sequence", absl::StrCat(frame->sequence));
  response.set_header("X-Sim-Time", jsonNumber(frame->simTime));
  response.set_header("X-Frame-Age-Ms",
                      absl::StrCat(frameAgeMilliseconds(*frame)));
  response.set_header("Cache-Control", "no-store");
  const size_t size = frame->jpeg.size();
  // cpp-httplib calls the provider until `size` bytes are written, each call
  // after its writability check.
  response.set_content_provider(
      size, "image/jpeg",
      [this, jpeg = std::move(frame->jpeg)](size_t offset, size_t length,
                                            httplib::DataSink& sink) {
        return !stopped() && sink.write(jpeg.data() + offset,
                                        std::min(length, kWritePieceBytes));
      });
}

bool WebViewerServer::Core::writeStreamPart(int64_t& lastSequence,
                                            httplib::DataSink& sink) const {
  const std::optional<EncodedFrame> frame =
      waitForFrame(lastSequence, kStreamPartWait);
  if (stopped()) return false;
  // Nothing new: cpp-httplib checks that the client is still there and asks
  // again.
  if (!frame.has_value()) return true;
  lastSequence = frame->sequence;
  const std::string header = absl::StrCat(
      "--", kStreamBoundary,
      "\r\nContent-Type: image/jpeg\r\nContent-Length: ", frame->jpeg.size(),
      "\r\nX-Frame-Sequence: ", frame->sequence,
      "\r\nX-Sim-Time: ", jsonNumber(frame->simTime), "\r\n\r\n");
  return writeInPieces(header, sink) && writeInPieces(frame->jpeg, sink) &&
         writeInPieces(kPartEnd, sink);
}

bool WebViewerServer::Core::writeInPieces(absl::string_view bytes,
                                          httplib::DataSink& sink) const {
  while (!bytes.empty()) {
    if (stopped()) return false;
    const size_t piece = std::min(bytes.size(), kWritePieceBytes);
    if (!sink.write(bytes.data(), piece)) return false;
    bytes.remove_prefix(piece);
  }
  return true;
}

void WebViewerServer::Core::serveStream(httplib::Response& response) const {
  if (stopped()) {
    answerUnavailable(response);
    return;
  }
  if (!mailbox_->tryAcquireClient()) {
    answer(response, kServiceUnavailable,
           absl::StrCat("too many viewers (", mailbox_->maxClients(),
                        "): close another viewer tab or stream"));
    return;
  }
  mailbox_->noteFrameRequest();
  openStreams_.fetch_add(1);
  response.set_header("Cache-Control", "no-store");
  // The releaser runs once, when cpp-httplib destroys the response: after the
  // stream ends, the client disconnects or the request is a HEAD that never
  // reads the provider.
  response.set_chunked_content_provider(
      absl::StrCat("multipart/x-mixed-replace; boundary=", kStreamBoundary),
      [this, lastSequence = int64_t{0}](size_t /*offset*/,
                                        httplib::DataSink& sink) mutable {
        return writeStreamPart(lastSequence, sink);
      },
      [this](bool /*success*/) {
        openStreams_.fetch_sub(1);
        mailbox_->releaseClient();
      });
}

void WebViewerServer::Core::serveState(httplib::Response& response) const {
  const ViewerStatus status = mailbox_->status();
  const std::optional<EncodedFrame> latest = mailbox_->latestFrameMetadata();
  std::string frame = "null";
  if (latest.has_value()) {
    frame = absl::StrCat(R"({"sequence":)", latest->sequence, R"(,"simTime":)",
                         jsonNumber(latest->simTime), R"(,"width":)",
                         latest->width, R"(,"height":)", latest->height,
                         R"(,"ageMs":)", frameAgeMilliseconds(*latest), "}");
  }
  response.set_header("Cache-Control", "no-store");
  response.set_content(
      absl::StrCat(
          R"({"clients":)", mailbox_->activeClients(), R"(,"viewers":)",
          mailbox_->viewerCount() + openStreams_.load(), R"(,"maxClients":)",
          mailbox_->maxClients(), R"(,"droppedInputs":)",
          inputs_->droppedCount(), R"(,"frame":)", frame, R"(,"viewer":)",
          viewerStatusJson(status), "}"),
      "application/json");
}

void WebViewerServer::Core::serveInput(const httplib::Request& request,
                                       httplib::Response& response) const {
  absl::StatusOr<absl::flat_hash_map<std::string, std::string>> parameters =
      singleParameters(request);
  if (!parameters.ok()) {
    answer(response, kBadRequest, parameters.status().message());
    return;
  }
  absl::StatusOr<ViewerInput> input =
      parseViewerInput(*parameters, options_.visualizationNames);
  if (!input.ok()) {
    answer(response, kBadRequest, input.status().message());
    return;
  }
  inputs_->push(*std::move(input));
  response.status = kNoContent;
}

void WebViewerServer::Core::serveSettings(const httplib::Request& request,
                                          httplib::Response& response) const {
  absl::StatusOr<absl::flat_hash_map<std::string, std::string>> parameters =
      singleParameters(request);
  if (!parameters.ok()) {
    answer(response, kBadRequest, parameters.status().message());
    return;
  }
  absl::StatusOr<ViewerInput> settings = parseViewerSettings(*parameters);
  if (!settings.ok()) {
    answer(response, kBadRequest, settings.status().message());
    return;
  }
  inputs_->push(*std::move(settings));
  response.status = kNoContent;
}

absl::StatusOr<std::unique_ptr<WebViewerServer>> WebViewerServer::Create(
    WebViewerServerOptions options,
    ViewerFrameMailbox* absl_nonnull mailbox,
    ViewerInputQueue* absl_nonnull inputs) {
  absl::StatusOr<BindAddress> address = parseBindAddress(options.address);
  if (!address.ok()) return address.status();
  absl::Status valid = checkOptions(options, *mailbox);
  if (!valid.ok()) return valid;
  if (!address->isLoopback) {
    LOG(WARNING)
        << "[WebViewerServer] binding " << address->canonical << ":"
        << options.port
        << ": anyone who can reach this address can watch the simulation and "
           "move its camera, and can deny the viewer (not "
           "the simulation) by holding its request threads; there is no "
           "authentication or TLS (robot_runtime/web_viewer/README.md, "
           "\"Trust model\")";
  }
  absl::StatusOr<std::unique_ptr<Core>> core =
      Core::Create(std::move(options), *std::move(address), mailbox, inputs);
  if (!core.ok()) return core.status();
  std::unique_ptr<WebViewerServer> server =
      absl::WrapUnique(new WebViewerServer(*std::move(core)));
  // LINT.IfChange(web_viewer_announcement)
  LOG(INFO) << "[WebViewerServer] serving the MuJoCo viewer at "
            << server->url();
  // LINT.ThenChange(//tools/deploy/session.sh:web_viewer_announcement)
  return server;
}

WebViewerServer::WebViewerServer(std::unique_ptr<Core> core)
    : core_(std::move(core)) {}

WebViewerServer::~WebViewerServer() = default;

int WebViewerServer::port() const {
  return core_->port();
}

std::string WebViewerServer::url() const {
  return core_->url();
}

}  // namespace robot::web_viewer
