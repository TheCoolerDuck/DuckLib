

namespace Duck.Modules
{
    public abstract class Module(Module? parent, string name)
    {
        private readonly string _name = name;
        public string name => (parent != null ? parent.name + "|" : "") + _name;
        public Module? parent = parent;
        public abstract Matrix Forward(Matrix m);
        public abstract Matrix[] GetParameters();
    }
}
