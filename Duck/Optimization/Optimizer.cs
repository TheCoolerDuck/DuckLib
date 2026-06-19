namespace Duck.Optimization
{
    public abstract class Optimizer(Matrix[] parameters)
    {
        public Matrix[] parameters = parameters;
        public abstract void step();
    }
}
