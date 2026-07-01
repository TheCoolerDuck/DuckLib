using System;
using System.Threading.Tasks;

namespace Duck.Management
{
    internal static class CPUManager
    {
        public static readonly int MinParallelChunks = Environment.ProcessorCount * 2;
        public const int TasksPerThread = 128;

        public static void RunTask(int start, int finish, Action<int> body, int tasksPerThread = TasksPerThread)
        {
            int length = finish - start;
            int n = length / tasksPerThread + (length % tasksPerThread == 0 ? 0 : 1);

            if (n < MinParallelChunks)
            {
                for (int i = 0; i < n; i++)
                    ExecuteChunk1D(i, start, length, tasksPerThread, body);
            }
            else
            {
                Parallel.For(0, n, i => ExecuteChunk1D(i, start, length, tasksPerThread, body));
            }
        }

        public static void RunTask(int startX, int finishX, int startY, int finishY, Action<int, int> body, int tasksPerThread = TasksPerThread)
        {
            int xd = finishX - startX;
            int yd = finishY - startY;
            int length = xd * yd;
            int n = length / tasksPerThread + (length % tasksPerThread == 0 ? 0 : 1);

            if (n < MinParallelChunks)
            {
                for (int i = 0; i < n; i++)
                    ExecuteChunk2D(i, startX, startY, xd, length, tasksPerThread, body);
            }
            else
            {
                Parallel.For(0, n, i => ExecuteChunk2D(i, startX, startY, xd, length, tasksPerThread, body));
            }
        }

        private static void ExecuteChunk1D(int i, int start, int length, int tasksPerThread, Action<int> body)
        {
            int s = i * tasksPerThread;
            int f = Math.Min(s + tasksPerThread, length);
            for (int k = s; k < f; k++)
                body(k + start);
        }

        private static void ExecuteChunk2D(int i, int startX, int startY, int xd, int length, int tasksPerThread, Action<int, int> body)
        {
            int s = i * tasksPerThread;
            int f = Math.Min(s + tasksPerThread, length);
            for (int k = s; k < f; k++)
                body(k % xd + startX, k / xd + startY);
        }
    }
}