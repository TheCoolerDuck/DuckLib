using Duck.Functions.Parameters;

namespace Duck.Modules
{
    public interface IModule
    {
        public Matrix Forward(Matrix m);
        public Matrix[] GetParameters();
    }
}
