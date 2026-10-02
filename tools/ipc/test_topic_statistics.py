"""Tests for topic_statistics.py: what `hz` and `list` compute from receive times and payload sizes."""

import unittest

import topic_statistics


def feed(
    statistics: topic_statistics.RateStatistics, times: list, size: int = 100
) -> None:
    for receive_time in times:
        statistics.add(receive_time, size)


class RateStatisticsTest(unittest.TestCase):

    def test_a_steady_stream_has_its_rate_and_no_jitter(self) -> None:
        period = 0.004
        statistics = topic_statistics.RateStatistics(window_size=50)
        feed(statistics, [10.0 + period * i for i in range(50)])
        snapshot = statistics.snapshot()
        self.assertIsNotNone(snapshot)
        self.assertAlmostEqual(snapshot.rate_hz, 1.0 / period, places=6)
        self.assertAlmostEqual(snapshot.mean_period_s, period, places=12)
        self.assertAlmostEqual(snapshot.min_period_s, period, places=12)
        self.assertAlmostEqual(snapshot.max_period_s, period, places=12)
        self.assertAlmostEqual(snapshot.jitter_s, 0.0, places=12)

    def test_jitter_is_the_standard_deviation_of_the_period(self) -> None:
        statistics = topic_statistics.RateStatistics(window_size=10)
        # Periods alternate between 1 and 3: mean 2, standard deviation 1.
        feed(statistics, [0.0, 1.0, 4.0, 5.0, 8.0])
        snapshot = statistics.snapshot()
        self.assertAlmostEqual(snapshot.mean_period_s, 2.0)
        self.assertAlmostEqual(snapshot.jitter_s, 1.0)
        self.assertEqual(snapshot.min_period_s, 1.0)
        self.assertEqual(snapshot.max_period_s, 3.0)

    def test_the_window_forgets_old_messages(self) -> None:
        statistics = topic_statistics.RateStatistics(window_size=3)
        # A slow start, then three messages 0.1 s apart: only the last three count.
        feed(statistics, [0.0, 5.0, 10.0, 10.1, 10.2])
        snapshot = statistics.snapshot()
        self.assertEqual(snapshot.window_count, 3)
        self.assertEqual(statistics.total_count, 5)
        self.assertAlmostEqual(snapshot.rate_hz, 10.0)

    def test_sizes_and_bandwidth(self) -> None:
        statistics = topic_statistics.RateStatistics(window_size=10)
        for receive_time, size in [(0.0, 100), (1.0, 300), (2.0, 200)]:
            statistics.add(receive_time, size)
        snapshot = statistics.snapshot()
        self.assertEqual(snapshot.min_size_bytes, 100)
        self.assertEqual(snapshot.max_size_bytes, 300)
        self.assertAlmostEqual(snapshot.mean_size_bytes, 200.0)
        # 500 bytes arrived over the 2 s after the first message.
        self.assertAlmostEqual(snapshot.bandwidth_bytes_per_s, 250.0)

    def test_no_snapshot_without_two_distinct_receive_times(self) -> None:
        statistics = topic_statistics.RateStatistics(window_size=10)
        self.assertIsNone(statistics.snapshot())
        statistics.add(1.0, 10)
        self.assertIsNone(statistics.snapshot())
        statistics.add(1.0, 10)
        self.assertIsNone(statistics.snapshot())

    def test_the_window_holds_at_least_two_messages(self) -> None:
        with self.assertRaises(ValueError):
            topic_statistics.RateStatistics(window_size=1)

    def test_the_report_names_the_topic_rate_jitter_and_size(self) -> None:
        statistics = topic_statistics.RateStatistics(window_size=10)
        feed(statistics, [0.0, 0.5, 1.0], size=2048)
        report = topic_statistics.format_snapshot("mpc/status", statistics.snapshot())
        self.assertIn("mpc/status: average rate 2.000 Hz", report)
        self.assertIn("jitter (std dev) 0.000 ms", report)
        self.assertIn("size mean 2.0 KiB", report)


class FormatBytesTest(unittest.TestCase):

    def test_units(self) -> None:
        self.assertEqual(topic_statistics.format_bytes(512), "512 B")
        self.assertEqual(topic_statistics.format_bytes(40 * 1024), "40.0 KiB")
        self.assertEqual(topic_statistics.format_bytes(3 * 1024 * 1024), "3.0 MiB")


class TopicCensusTest(unittest.TestCase):

    def test_summaries_are_sorted_with_counts_types_and_rates(self) -> None:
        census = topic_statistics.TopicCensus()
        for receive_time in (0.0, 0.5, 1.0):
            census.add(
                "robot/state", "humanoid_mpc_msgs.RobotStateSample", receive_time
            )
        census.add("mpc/status", "humanoid_mpc_msgs.MpcStatus", 0.2)
        summaries = census.summaries()
        self.assertEqual(
            [summary.topic for summary in summaries], ["mpc/status", "robot/state"]
        )
        self.assertEqual(summaries[1].count, 3)
        self.assertAlmostEqual(summaries[1].rate_hz, 2.0)
        # One message gives no rate.
        self.assertIsNone(summaries[0].rate_hz)

    def test_a_topic_with_two_types_lists_both(self) -> None:
        census = topic_statistics.TopicCensus()
        census.add("operator/fsm_command", "humanoid_mpc_msgs.FsmCommand", 0.0)
        census.add("operator/fsm_command", "humanoid_mpc_msgs.YamlDocument", 0.1)
        table = topic_statistics.format_census(census.summaries())
        self.assertIn(
            "humanoid_mpc_msgs.FsmCommand, humanoid_mpc_msgs.YamlDocument", table
        )

    def test_the_table_has_a_header_and_one_line_per_topic(self) -> None:
        census = topic_statistics.TopicCensus()
        census.add("a", "T", 0.0)
        census.add("b", "U", 0.0)
        lines = topic_statistics.format_census(census.summaries()).splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("TOPIC"))
        self.assertTrue(lines[1].startswith("a "))


if __name__ == "__main__":
    unittest.main()
