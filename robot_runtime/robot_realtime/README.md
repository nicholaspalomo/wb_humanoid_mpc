# robot_realtime

The building blocks of the robot process's **realtime thread** (see
[`humanoid_nmpc/docs/distributed_runtime/README.md`](../../humanoid_nmpc/docs/distributed_runtime/README.md), "The
realtime thread"). The library is `//robot_runtime/robot_realtime`, the namespace is `robot::realtime`, and it depends
only on Abseil: no ROS, no protobuf and nothing from `humanoid_nmpc`.

| Header | What it gives the realtime thread |
|---|---|
| `SpscQueue.h` | `SpscQueue<T>`: a bounded, lock-free single-producer / single-consumer ring with preallocated slots, for events and telemetry samples |
| `RealtimeThread.h` | `setCurrentThreadRealtime`, `lockProcessMemory`, `prefaultStack`, `setCurrentThreadAffinity`, `setCurrentThreadName`, and `configureCurrentThread` to apply them all at once |
| `PeriodicTimer.h` | `PeriodicTimer`: absolute-deadline pacing on `CLOCK_MONOTONIC` with a selectable `OverrunPolicy` |
| `LoopTimingStats.h` | `LoopTimingStats`: period, compute time, overruns and wake-up latency over a reporting window |
| `LoopTimingSnapshot.h` | `LoopTimingSnapshot`: the plain struct a window condenses into, read by the communication thread |

For latest-value data, such as the observation or the timing snapshot, use `robot::TripleBuffer`
(`robot_runtime/robot_core`).

## The realtime rules, and how each part keeps them

Once constructed, nothing here allocates, takes a lock, does I/O or logs. `:test_realtime_allocations` checks the
allocation part by counting every `malloc` of the process.

- **`SpscQueue<T>(capacity, prototype)`** constructs every slot as a copy of `prototype`. `tryPush()` copy-assigns into
  a slot and `tryPop()` copy-assigns out of one. So a payload with heap storage, such as an `Eigen::VectorXd` or a
  `std::vector`, passes through without allocating as long as it keeps the prototype's size. A push into a full queue
  returns `false` and is counted in `droppedCount()`, which feeds `LoopTiming.telemetry_samples_dropped`. The producer
  never waits for the consumer. `tryPushInPlace()` and `tryPopInPlace()` write into and read from the slot directly,
  which saves a copy when the sample is large.
- **`PeriodicTimer`** sleeps with `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)` towards deadlines spaced
  `period` apart, so neither compute time nor wake-up latency adds up to drift. `waitForNextPeriod()` returns a
  `TimerWakeup`: the deadline it served, how late the thread woke, and how many deadlines it skipped. When a cycle
  overruns:
  - `OverrunPolicy::kSkipMissedPeriods` drops the deadlines that have passed and stays on the original grid. A
    controller should use this.
  - `OverrunPolicy::kCatchUp` runs the missed cycles back to back. A loop that counts cycles as time, such as a
    fixed-step simulation, should use this.
- **`LoopTimingStats`** accumulates integer nanoseconds. When a reporting window closes, it condenses them into a
  `LoopTimingSnapshot`. The snapshot is trivially copyable, so it goes to the communication thread through a
  `robot::TripleBuffer<LoopTimingSnapshot>`. `LoopTimingSnapshot.h` shows which snapshot field fills which
  `humanoid_mpc_msgs.LoopTiming` field. The communication thread does that conversion, so this package stays free of
  protobuf.

```cpp
// Once, on the realtime thread, before the loop.
const absl::Status setup = robot::realtime::configureCurrentThread(
    {.name = "rt_control", .priority = absl::GetFlag(FLAGS_realtime_priority), .cores = mrtCores});
if (!setup.ok()) LOG(WARNING) << "running without realtime scheduling: " << setup;

robot::realtime::PeriodicTimer timer(std::chrono::milliseconds(2), robot::realtime::OverrunPolicy::kSkipMissedPeriods);
robot::realtime::LoopTimingStats stats(timer.period(), std::chrono::seconds(1));
timer.start();
while (running.load(std::memory_order_relaxed)) {
  const robot::realtime::TimerWakeup wakeup = timer.waitForNextPeriod();
  // ... read the backend, run the controller, write the backend ...
  telemetry.tryPushInPlace([&](TelemetrySample& slot) { /* fill the slot */ });
  if (stats.addCycle(wakeup, robot::realtime::monotonicNow())) {
    timingBuffer.writeSlot() = stats.lastSnapshot();
    timingBuffer.publishWrite();
  }
}
```

## Privileges

`setCurrentThreadRealtime()` and `lockProcessMemory()` return an error, never abort, when the process lacks the
privilege. `PermissionDenied` or `ResourceExhausted` comes with a message that says what to grant:

- `SCHED_FIFO` needs `CAP_SYS_NICE`, or an `RLIMIT_RTPRIO` at least as high as the priority: `ulimit -r`,
  `/etc/security/limits.conf`, or `docker run --cap-add=SYS_NICE --ulimit rtprio=99`.
- `mlockall` needs `CAP_IPC_LOCK`, or an `RLIMIT_MEMLOCK` larger than the process: `ulimit -l unlimited`, or
  `docker run --ulimit memlock=-1`.

The dev container grants neither, so simulation runs without them. `configureCurrentThread()` with `priority = 0`
sets only the name and the affinity, and never needs a privilege.

`configureCurrentThread()` tries every step, whatever the steps before it did: a thread the kernel pinned to only some
of its cores, or whose process may not lock its memory, still runs `SCHED_FIFO` when it may. Its error carries the
code of the first step that failed and names every step that failed. `lockProcessMemory()` is process-wide, and
`MCL_FUTURE` makes every later allocation of the process fail once its locked memory would pass a finite
`RLIMIT_MEMLOCK`. A thread of a process that allocates freely (the MPC solver) therefore sets
`memoryLock = MemoryLock::kNone`, which gives it the scheduling without locking or prefaulting anything.

## Tests

`bazel test //robot_runtime/robot_realtime/...`

| Target | What it pins |
|---|---|
| `:test_spsc_queue` | FIFO order, drops counted when full, slots start as the prototype, and two-thread stress runs: no loss and no reordering when the producer retries, `received + dropped == pushed` when it does not, no torn payloads |
| `:test_periodic_timer` | the deadline arithmetic of both overrun policies, exactly. Against the clock: no wake-up before its deadline, deadlines on the grid, no drift when half of each period is compute, and how each policy behaves after a real overrun |
| `:test_loop_timing_stats` | window boundaries, windows that tile time and count each cycle once, overruns, jitter extremes, and snapshots that reach another thread intact through a `TripleBuffer` |
| `:test_realtime_thread` | name, affinity, priority checks, `SCHED_FIFO` that either takes effect or reports `PermissionDenied`, and the unprivileged path forced in a child process that drops `CAP_SYS_NICE` and `CAP_IPC_LOCK` and lowers its limits. That `configureCurrentThread()` carries on past a failed step and names every failure, and that `MemoryLock::kNone` touches no memory. Also that a prefaulted stack takes no page faults |
| `:test_realtime_allocations` | no heap allocation in `tryPush`/`tryPop`, `waitForNextPeriod`, `addCycle` or a whole realtime cycle |

`:allocation_counter` (testonly) interposes glibc's `malloc`, `calloc`, `realloc`, `aligned_alloc`, `memalign` and
`posix_memalign` to count every heap allocation of a test binary. It began as a copy of the state estimator's counter
(`humanoid_state_estimation/humanoid_state_estimator/test/AllocationCounter.h`) and also counts aligned allocations.
A binary can link only one of the two. `heapAllocationCount()` counts the whole process;
`heapAllocationCountOnThisThread()` counts the calling thread's allocations alone, for a call on a realtime thread while
other threads of the process (a bus's IO thread, a solver) allocate beside it, as
`//humanoid_nmpc/humanoid_mpc_ipc:test_remote_mpc_link_allocations` does.
