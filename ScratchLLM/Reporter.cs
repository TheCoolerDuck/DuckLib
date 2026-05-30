
using System.Diagnostics;

/// <summary>
/// Configuration container for the Reporter system.
/// Controls how frequently reports are published.
/// </summary>
struct ReporterSettings
{
    /// <summary>
    /// Interval between report publications, expressed in seconds.
    /// Internally stored as milliseconds for precision and consistency.
    /// </summary>
    public double secondsBetweenReports
    {
        readonly get
        {
            return millisecondsBetweenReports / 1000.0;
        }
        set
        {
            millisecondsBetweenReports = (long)(value * 1000);
        }
    }

    /// <summary>
    /// Interval between report publications in milliseconds.
    /// Defaults to 60 seconds.
    /// </summary>
    public long millisecondsBetweenReports = 60_000;

    public ReporterSettings() { }
}

/// <summary>
/// Thread-safe reporting utility that aggregates structured reports
/// and periodically publishes them to the console.
/// </summary>
internal static class Reporter
{
    /// <summary>
    /// Represents a single report entry.
    /// Reports may optionally form a hierarchy via parentSubject.
    /// </summary>
    public struct Report(string subject, string information, string? parentSubject = null)
    {
        public string? parentSubject = parentSubject;
        public string subject = subject;
        public string information = information;
        public TimeSpan at = watch.Elapsed;
    }

    /// <summary>
    /// Defines how Update behaves when another thread currently holds the lock.
    /// </summary>
    public enum ThreadCollisionAction
    {
        /// <summary>
        /// Block until the lock is acquired.
        /// </summary>
        wait,

        /// <summary>
        /// Skip the update if the lock cannot be acquired immediately.
        /// </summary>
        skip
    }

    /// <summary>
    /// Global stopwatch used for consistent timestamps across reports.
    /// </summary>
    private static readonly Stopwatch watch = Stopwatch.StartNew();

    /// <summary>
    /// Stores the most recent report per subject.
    /// Subject acts as a unique key.
    /// </summary>
    private static readonly Dictionary<string, Report> reports = [];

    /// <summary>
    /// Incrementing identifier for published report batches.
    /// </summary>
    private static long reportID = 1;

    /// <summary>
    /// Synchronization object for thread-safe access to the reports dictionary.
    /// </summary>
    private static readonly object threadLock = new();

    /// <summary>
    /// Signals the background reporting thread to terminate.
    /// </summary>
    private static bool end = false;

    /// <summary>
    /// Global settings instance controlling report behavior.
    /// </summary>
    public static ReporterSettings settings = new();

    /// <summary>
    /// Static constructor starts the background publishing loop.
    /// </summary>
    static Reporter()
    {
        new Thread(MainLoop)
        {
            IsBackground = true
        }.Start();
    }

    /// <summary>
    /// Background loop that waits until the next scheduled publish time
    /// and then emits accumulated reports.
    /// </summary>
    private static void MainLoop()
    {
        long next = settings.millisecondsBetweenReports;

        while (!end)
        {
            // Busy-wait with sleep granularity to reduce CPU usage
            while (next > watch.ElapsedMilliseconds)
            {
                Thread.Sleep(100);

                if (end)
                    break;
            }

            next += settings.millisecondsBetweenReports;

            Publish();
        }

        Console.WriteLine($"[{watch.Elapsed:hh\\:mm\\:ss}] Reporter ended");
    }

    /// <summary>
    /// Add or update a report entry.
    /// If a report with the same subject already exists, it is replaced.
    /// </summary>
    public static void Update(Report report, ThreadCollisionAction threadCollisionAction = ThreadCollisionAction.skip)
    {
        if (threadCollisionAction == ThreadCollisionAction.wait)
        {
            lock (threadLock)
            {
                reports[report.subject] = report;
            }
        }
        else
        {
            if (!Monitor.TryEnter(threadLock))
                return;

            try
            {
                reports[report.subject] = report;
            }
            finally
            {
                Monitor.Exit(threadLock);
            }
        }
    }

    /// <summary>
    /// Signal the reporter to shut down.
    /// The background thread will exit cleanly.
    /// </summary>
    public static void End()
    {
        end = true;
    }

    /// <summary>
    /// Publish all collected reports to the console in a hierarchical format.
    /// Clears the report buffer after publishing.
    /// </summary>
    public static void Publish()
    {
        if (reports.Count == 0)
            return;

        int indent = 0;

        const string lineStarts = " -*|";

        WriteArticle(null);

        /// <summary>
        /// Recursively writes a report and all of its children,
        /// preserving temporal order and visual hierarchy.
        /// </summary>
        void WriteArticle(Report? report)
        {
            if (report == null)
            {
                Console.WriteLine($"[{watch.Elapsed:hh\\:mm\\:ss}] Report {reportID++}:");
            }
            else
            {
                string prefix = string.Concat(Enumerable.Repeat("  ", indent));
                string connector = indent == 0 ? "" : "" + lineStarts[(indent - 1) % lineStarts.Length];

                Console.WriteLine($"{prefix}{connector}[{report.Value.at:hh\\:mm\\:ss}] {report.Value.subject}: {report.Value.information}");
            }

            indent++;

            foreach (Report r in reports.Values.Where(r => r.parentSubject == report?.subject).OrderBy(r => r.at))
            {
                WriteArticle(r);
            }

            indent--;
        }

        reports.Clear();
    }
}
