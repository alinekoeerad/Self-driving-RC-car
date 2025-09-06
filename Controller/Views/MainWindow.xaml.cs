using Controller.ViewModels;
using System.Windows;
using System.Windows.Input;


namespace Controller
{
    public partial class MainWindow : Window
    {
        private readonly MainWindowViewModel _viewModel;

        public MainWindow()
        {
            InitializeComponent();
            _viewModel = (MainWindowViewModel)DataContext;
            this.Closing += (s, e) => _viewModel.Cleanup();
        }

        // ADDED: This method captures mouse clicks on the video area
        // and passes the necessary information to the ViewModel for processing.
        private void Canvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            // The name "LiveVideoImage" was added to the Image tag in the XAML.
            var imageControl = LiveVideoImage;
            if (imageControl != null && _viewModel != null)
            {
                var clickPosition = e.GetPosition(imageControl);
                _viewModel.HandleImageClick(clickPosition, imageControl.ActualWidth, imageControl.ActualHeight);
            }
        }
    }
}