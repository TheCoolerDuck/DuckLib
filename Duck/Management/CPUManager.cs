using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Duck.Management
{
    internal static class CPUManager
    {
        public const int tasksPerThread = 128;

        public static void RunTask(int start, int finish, Action<int> body, int tasksPerThread = tasksPerThread)
        {
            int length = finish - start;
            int n = length / tasksPerThread + (length % tasksPerThread == 0 ? 0 : 1);
            Parallel.For(0, n, i =>
            {
                int s = i * tasksPerThread;
                int f = Math.Min(s + tasksPerThread, length);
                for (int k = s; k < f; k++)
                {
                    body(k + start);
                }
            });
        }

        public static void RunTask(int startX, int finishX, int startY, int finishY, Action<int, int> body, int tasksPerThread = tasksPerThread)
        {
            int xd = finishX - startX;
            int yd = finishY - startY;
            int length = xd * yd;
            int n = length / tasksPerThread + (length % tasksPerThread == 0 ? 0 : 1);
            Parallel.For(0, n, i =>
            {
                int start = i * tasksPerThread;
                int end = Math.Min(start + tasksPerThread, length);
                for (int k = start; k < end; k++)
                {
                    int x = k % xd + startX;
                    int y = k / xd + startY;
                    body(x, y);
                }
            });
        }
    }
}
