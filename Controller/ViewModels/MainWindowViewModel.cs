// In Controller/ViewModels/MainWindowViewModel.cs

using Controller.Common;
using Controller.Models;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media.Imaging;

namespace Controller.ViewModels
{
    public class MainWindowViewModel : ViewModelBase
    {
        // Collections for UI binding
        public ObservableCollection<NodeViewModel> Nodes { get; } = new ObservableCollection<NodeViewModel>();
        public ObservableCollection<EdgeViewModel> Edges { get; } = new ObservableCollection<EdgeViewModel>();
        public ObservableCollection<LogEntryViewModel> LogEntries { get; } = new ObservableCollection<LogEntryViewModel>();

        // Status properties
        private string _connectionStatus = "Disconnected";
        public string ConnectionStatus { get => _connectionStatus; set { _connectionStatus = value; OnPropertyChanged(); } }

        private NodeViewModel? _selectedTargetNode;
        public NodeViewModel? SelectedTargetNode { get => _selectedTargetNode; set { _selectedTargetNode = value; OnPropertyChanged(); } }

        private BitmapImage? _liveVideoFrame;
        public BitmapImage? LiveVideoFrame { get => _liveVideoFrame; set { _liveVideoFrame = value; OnPropertyChanged(); } }

        // ADDED: Property to store the robot's current heading for the UI
        private string _robotHeading = "none";
        public string RobotHeading { get => _robotHeading; set { _robotHeading = value; OnPropertyChanged(); } }

        // Setup Mode Properties
        public ObservableCollection<Point> PerspectivePoints { get; } = new ObservableCollection<Point>();
        private bool _isInCalibrationMode;
        public bool IsInCalibrationMode { get => _isInCalibrationMode; set { _isInCalibrationMode = value; OnPropertyChanged(); OnPropertyChanged(nameof(IsInSetupMode)); } }

        private bool _isInPerspectiveSetupMode;
        public bool IsInPerspectiveSetupMode { get => _isInPerspectiveSetupMode; set { _isInPerspectiveSetupMode = value; OnPropertyChanged(); OnPropertyChanged(nameof(IsInSetupMode)); } }

        public bool IsInSetupMode => IsInCalibrationMode || IsInPerspectiveSetupMode;
        public string PerspectiveSetupInstructions => $"Click {4 - PerspectivePoints.Count} more point(s) on the video feed.";

        // Commands
        public ICommand StartExplorationCommand { get; }
        public ICommand EmergencyStopCommand { get; }
        public ICommand ResetMapCommand { get; }
        public ICommand StartCalibrationCommand { get; }
        public ICommand CaptureCalibrationImageCommand { get; }
        public ICommand FinishCalibrationCommand { get; }
        public ICommand StartPerspectiveSetupCommand { get; }
        public ICommand FinishPerspectiveSetupCommand { get; }
        public ICommand CancelSetupCommand { get; }

        private bool _isLedOn;
        public bool IsLedOn
        {
            get => _isLedOn;
            set
            {
                if (_isLedOn != value)
                {
                    _isLedOn = value;
                    OnPropertyChanged();
                    _ = _communicator.SendCommandAsync("toggle_led", new { state = _isLedOn });
                }
            }
        }

        private readonly object _lock = new object();
        private readonly PythonCommunicator _communicator;

        public MainWindowViewModel()
        {
            _communicator = new PythonCommunicator();
            _communicator.OnDataReceived += ProcessIncomingData;
            _communicator.OnLogMessage += AddLogEntry;
            _ = _communicator.ConnectAsync();

            StartExplorationCommand = new RelayCommand(async _ => await _communicator.SendCommandAsync("start_exploration"));
            EmergencyStopCommand = new RelayCommand(async _ => await _communicator.SendCommandAsync("emergency_stop"));
            ResetMapCommand = new RelayCommand(async _ =>
            {
                await _communicator.SendCommandAsync("reset_map");
                Nodes.Clear();
                Edges.Clear();
            });

            StartCalibrationCommand = new RelayCommand(async _ => {
                await _communicator.SendCommandAsync("start_calibration");
                IsInCalibrationMode = true;
            });
            CaptureCalibrationImageCommand = new RelayCommand(async _ => await _communicator.SendCommandAsync("capture_calib_image"));
            FinishCalibrationCommand = new RelayCommand(async _ => {
                await _communicator.SendCommandAsync("finish_calib");
                IsInCalibrationMode = false;
            });

            StartPerspectiveSetupCommand = new RelayCommand(async _ => {
                await _communicator.SendCommandAsync("start_perspective_setup");
                PerspectivePoints.Clear();
                IsInPerspectiveSetupMode = true;
                OnPropertyChanged(nameof(PerspectiveSetupInstructions));
            });

            FinishPerspectiveSetupCommand = new RelayCommand(
                async _ => {
                    await _communicator.SendCommandAsync("finish_perspective");
                    IsInPerspectiveSetupMode = false;
                },
                _ => PerspectivePoints.Count == 4
            );

            CancelSetupCommand = new RelayCommand(async _ => {
                await _communicator.SendCommandAsync("cancel_setup");
                IsInCalibrationMode = false;
                IsInPerspectiveSetupMode = false;
            });
        }

        public void HandleImageClick(Point clickPos, double actualWidth, double actualHeight)
        {
            if (!IsInPerspectiveSetupMode || PerspectivePoints.Count >= 4) return;
            if (actualWidth == 0 || actualHeight == 0) return;

            double scaledX = (clickPos.X / actualWidth) * 640;
            double scaledY = (clickPos.Y / actualHeight) * 480;

            PerspectivePoints.Add(clickPos);
            var scaledPointPayload = new { x = scaledX, y = scaledY };
            _ = _communicator.SendCommandAsync("perspective_point_clicked", scaledPointPayload);

            OnPropertyChanged(nameof(PerspectiveSetupInstructions));
            CommandManager.InvalidateRequerySuggested();
        }

        private void ProcessIncomingData(string jsonData)
        {
            try
            {
                var pythonData = JsonConvert.DeserializeObject<PythonData>(jsonData);
                if (pythonData == null) return;

                Application.Current.Dispatcher.Invoke(() =>
                {
                    lock (_lock)
                    {
                        if (!string.IsNullOrEmpty(pythonData.Log)) AddLogEntry(pythonData.Log);
                        if (!string.IsNullOrEmpty(pythonData.Image)) UpdateVideoFrame(pythonData.Image);

                        // UPDATED: Set the robot heading from the incoming data
                        RobotHeading = pythonData.RobotHeading;

                        UpdateStatus(pythonData.IsConnected ? "✅ Connected to Python Server" : "❌ Disconnected");

                        var existingNodes = Nodes.ToDictionary(n => n.Id);
                        foreach (var pyNode in pythonData.Nodes)
                        {
                            if (pyNode.Position == null || pyNode.Position.Count != 2) continue;

                            if (existingNodes.TryGetValue(pyNode.Id, out var existingNode))
                            {
                                // Store the raw, unscaled position from Python
                                existingNode.RawX = pyNode.Position[0];
                                existingNode.RawY = pyNode.Position[1];
                            }
                            else
                            {
                                var newNode = new NodeViewModel(pyNode.Id, this)
                                {
                                    RawX = pyNode.Position[0],
                                    RawY = pyNode.Position[1]
                                };
                                Nodes.Add(newNode);
                            }
                        }

                        // After all nodes have their raw positions, calculate display positions
                        RescaleAndCenterMap();

                        Edges.Clear();
                        foreach (var pyEdge in pythonData.Edges)
                        {
                            var fromNode = Nodes.FirstOrDefault(n => n.Id == pyEdge.From);
                            var toNode = Nodes.FirstOrDefault(n => n.Id == pyEdge.To);
                            if (fromNode != null && toNode != null)
                            {
                                Edges.Add(new EdgeViewModel
                                {
                                    StartPoint = new Point(fromNode.X, fromNode.Y),
                                    EndPoint = new Point(toNode.X, toNode.Y),
                                    StartNodeId = fromNode.Id,
                                    EndNodeId = toNode.Id
                                });
                            }
                        }

                        UpdateNodeColors(pythonData.CurrentNode);
                        UpdateEdgeColors(pythonData.NavigationPath);
                    }
                });
            }
            catch (Exception ex)
            {
                AddLogEntry($"⚠️ ERROR processing data: {ex.Message}");
            }
        }

        // Helper method for auto-fitting the map
        private void RescaleAndCenterMap()
        {
            if (Nodes.Count < 1) return;

            // 1. Define canvas size and padding
            double canvasWidth = 300; // Match the Height/Width of your Border in XAML
            double canvasHeight = 300;
            double padding = 20;

            // 2. Find the bounding box of the graph using raw coordinates
            double minX = Nodes.Min(n => n.RawX);
            double minY = Nodes.Min(n => n.RawY);
            double maxX = Nodes.Max(n => n.RawX);
            double maxY = Nodes.Max(n => n.RawY);

            double graphWidth = maxX - minX;
            double graphHeight = maxY - minY;

            // Handle the case of a single node or a straight line
            if (graphWidth < 1) graphWidth = 1;
            if (graphHeight < 1) graphHeight = 1;

            // 3. Calculate the scale factor to fit the graph inside the canvas
            double scaleX = (canvasWidth - padding * 2) / graphWidth;
            double scaleY = (canvasHeight - padding * 2) / graphHeight;
            double scale = Math.Min(scaleX, scaleY);

            // 4. Calculate offsets to center the graph
            double offsetX = (canvasWidth - graphWidth * scale) / 2;
            double offsetY = (canvasHeight - graphHeight * scale) / 2;

            // 5. Apply the new scaled and centered positions to each node's display properties (X, Y)
            foreach (var node in Nodes)
            {
                // Translate raw coordinate to origin (0,0), scale it, then apply the offset
                node.X = offsetX + ((node.RawX - minX) * scale);
                node.Y = offsetY + ((node.RawY - minY) * scale);
            }
        }

        private void UpdateVideoFrame(string base64Image)
        {
            try
            {
                byte[] imageBytes = Convert.FromBase64String(base64Image);
                using (var ms = new MemoryStream(imageBytes))
                {
                    var frame = new BitmapImage();
                    frame.BeginInit();
                    frame.CacheOption = BitmapCacheOption.OnLoad;
                    frame.StreamSource = ms;
                    frame.EndInit();
                    frame.Freeze();
                    LiveVideoFrame = frame;
                }
            }
            catch (Exception ex)
            {
                AddLogEntry($"⚠️ Image decoding failed: {ex.Message}");
            }
        }

        private void UpdateEdgeColors(List<string> path)
        {
            foreach (var edge in Edges)
            {
                edge.StrokeColor = System.Windows.Media.Brushes.Gray;
                edge.StrokeThickness = 2;
            }

            if (path == null || path.Count < 2) return;

            for (int i = 0; i < path.Count - 1; i++)
            {
                string fromId = path[i];
                string toId = path[i + 1];
                var edgeToHighlight = Edges.FirstOrDefault(e => (e.StartNodeId == fromId && e.EndNodeId == toId) || (e.StartNodeId == toId && e.EndNodeId == fromId));
                if (edgeToHighlight != null)
                {
                    edgeToHighlight.StrokeColor = System.Windows.Media.Brushes.LawnGreen;
                    edgeToHighlight.StrokeThickness = 4;
                }
            }
        }

        private void UpdateNodeColors(string? currentNodeId)
        {
            foreach (var node in Nodes)
            {
                if (node.Id == currentNodeId) node.FillColor = System.Windows.Media.Brushes.Red;
                else if (node == SelectedTargetNode) node.FillColor = System.Windows.Media.Brushes.Yellow;
                else node.FillColor = System.Windows.Media.Brushes.CornflowerBlue;
            }
        }

        public void SelectNodeAsTarget(NodeViewModel node)
        {
            lock (_lock)
            {
                SelectedTargetNode = node;
                UpdateNodeColors(null);
                AddLogEntry($"✅ Target set to Node: {node.Id}");
                var payload = new { target_node = node.Id };
                _ = _communicator.SendCommandAsync("set_target", payload);
            }
        }

        private void AddLogEntry(string message)
        {
            Application.Current.Dispatcher.Invoke(() =>
            {
                if (LogEntries.Count > 200) LogEntries.RemoveAt(0);
                LogEntries.Add(new LogEntryViewModel(message));
            });
        }

        private void UpdateStatus(string status)
        {
            ConnectionStatus = status;
        }

        public void Cleanup()
        {
            _communicator.Dispose();
        }
    }
}