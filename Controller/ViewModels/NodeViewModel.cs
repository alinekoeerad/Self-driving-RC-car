// In Controller/ViewModels/NodeViewModel.cs

using Controller.Common;
using System.Windows.Input;
using System.Windows.Media;

namespace Controller.ViewModels
{
    // Represents a visual node (ellipse) on the map canvas.
    public class NodeViewModel : ViewModelBase
    {
        private readonly MainWindowViewModel _mainVm;
        public string Id { get; }

        // --- NEW: Properties to store the original, unscaled coordinates from Python ---
        public double RawX { get; set; }
        public double RawY { get; set; }
        // --------------------------------------------------------------------------

        // These properties will now hold the final, scaled coordinates for display
        private double _x;
        public double X { get => _x; set { _x = value; OnPropertyChanged(); } }

        private double _y;
        public double Y { get => _y; set { _y = value; OnPropertyChanged(); } }

        private Brush _fillColor = Brushes.CornflowerBlue;
        public Brush FillColor { get => _fillColor; set { _fillColor = value; OnPropertyChanged(); } }

        public ICommand SelectNodeCommand { get; }

        public NodeViewModel(string id, MainWindowViewModel mainVm)
        {
            Id = id;
            _mainVm = mainVm;
            SelectNodeCommand = new RelayCommand(param => _mainVm.SelectNodeAsTarget(this));
        }
    }
}