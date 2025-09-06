using Controller.Common;
using System.Windows;
using System.Windows.Media;

namespace Controller.ViewModels
{
    // Represents a visual edge (line) on the map canvas.
    public class EdgeViewModel : ViewModelBase
    {
        // ADDED: IDs of the connected nodes for path highlighting logic
        public string StartNodeId { get; set; } = string.Empty;
        public string EndNodeId { get; set; } = string.Empty;

        private Point _startPoint;
        public Point StartPoint { get => _startPoint; set { _startPoint = value; OnPropertyChanged(); } }

        private Point _endPoint;
        public Point EndPoint { get => _endPoint; set { _endPoint = value; OnPropertyChanged(); } }

        private Brush _strokeColor = Brushes.Gray;
        public Brush StrokeColor { get => _strokeColor; set { _strokeColor = value; OnPropertyChanged(); } }

        private int _strokeThickness = 2;
        public int StrokeThickness { get => _strokeThickness; set { _strokeThickness = value; OnPropertyChanged(); } }
    }
}