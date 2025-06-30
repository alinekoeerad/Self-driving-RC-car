using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using Newtonsoft.Json;
using Point = System.Windows.Point;

namespace Controller
{
    public partial class MainWindow : Window
    {
        private readonly object _graphLock = new object();
        private GraphData _currentGraphData;
        private Thread _graphReceiverThread;
        private bool _isRunning = true;

        public MainWindow()
        {
            InitializeComponent();
            StartGraphReceiver();
            StartGraphUpdates();
        }

        private void StartGraphReceiver()
        {
            _graphReceiverThread = new Thread(() =>
            {
                TcpListener listener = new TcpListener(IPAddress.Parse("127.0.0.1"), 9998);
                listener.Start();

                while (_isRunning)
                {
                    try
                    {
                        using (TcpClient client = listener.AcceptTcpClient())
                        using (NetworkStream stream = client.GetStream())
                        {
                            byte[] buffer = new byte[16_384];
                            int bytesRead = stream.Read(buffer, 0, buffer.Length);
                            string json = System.Text.Encoding.UTF8.GetString(buffer, 0, bytesRead);

                            var pyData = JsonConvert.DeserializeObject<PythonGraphData>(json);
                            var graphData = ConvertToGraphData(pyData);

                            lock (_graphLock)
                            {
                                _currentGraphData = graphData;
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Dispatcher.Invoke(() =>
                            ErrorTextBlock.Text = $"Graph Error: {ex.Message}");
                    }
                }
            });
            _graphReceiverThread.IsBackground = true;
            _graphReceiverThread.Start();
        }

        private GraphData ConvertToGraphData(PythonGraphData pyData)
        {
            var nodes = new List<Node>();
            var edges = new List<Edge>();

            // Scale and position nodes
            const double scale = 40;
            const double offsetX = 100;
            const double offsetY = 100;

            foreach (var pyNode in pyData.nodes)
            {
                nodes.Add(new Node
                {
                    Id = pyNode.id,
                    Position = new Point(
                        pyNode.pos[0] * scale + offsetX,
                        pyNode.pos[1] * scale + offsetY)
                });
            }

            foreach (var pyEdge in pyData.edges)
            {
                var fromNode = pyData.nodes.First(n => n.id == pyEdge.from);
                var toNode = pyData.nodes.First(n => n.id == pyEdge.to);

                edges.Add(new Edge
                {
                    From = new Point(
                        fromNode.pos[0] * scale + offsetX,
                        fromNode.pos[1] * scale + offsetY),
                    To = new Point(
                        toNode.pos[0] * scale + offsetX,
                        toNode.pos[1] * scale + offsetY),
                    Weight = pyEdge.weight
                });
            }

            return new GraphData
            {
                Nodes = nodes,
                Edges = edges,
                CurrentNode = pyData.current_node?.ToString()
            };
        }

        private void StartGraphUpdates()
        {
            Thread thread = new Thread(() =>
            {
                while (_isRunning)
                {
                    if (_currentGraphData != null)
                    {
                        Dispatcher.Invoke(() => DrawGraphOnCanvas(_currentGraphData));
                    }
                    Thread.Sleep(200); // Update every 200ms
                }
            });
            thread.IsBackground = true;
            thread.Start();
        }

        private void DrawGraphOnCanvas(GraphData graphData)
        {
            GraphCanvas.Children.Clear();

            // Draw edges first
            foreach (var edge in graphData.Edges)
            {
                var line = new Line
                {
                    X1 = edge.From.X,
                    Y1 = edge.From.Y,
                    X2 = edge.To.X,
                    Y2 = edge.To.Y,
                    Stroke = Brushes.Black,
                    StrokeThickness = 1.5
                };
                GraphCanvas.Children.Add(line);

                // Edge weight label
                var midPoint = new Point(
                    (edge.From.X + edge.To.X) / 2,
                    (edge.From.Y + edge.To.Y) / 2);

                var label = new TextBlock
                {
                    Text = edge.Weight.ToString(),
                    Background = Brushes.White,
                    Foreground = Brushes.Black,
                    FontSize = 10
                };
                Canvas.SetLeft(label, midPoint.X);
                Canvas.SetTop(label, midPoint.Y);
                GraphCanvas.Children.Add(label);
            }

            // Draw nodes
            foreach (var node in graphData.Nodes)
            {
                var ellipse = new Ellipse
                {
                    Width = 20,
                    Height = 20,
                    Fill = Brushes.LightBlue,
                    Stroke = Brushes.Black,
                    StrokeThickness = 1
                };
                Canvas.SetLeft(ellipse, node.Position.X - 10);
                Canvas.SetTop(ellipse, node.Position.Y - 10);
                GraphCanvas.Children.Add(ellipse);

                // Node ID label
                var label = new TextBlock
                {
                    Text = node.Id,
                    FontWeight = FontWeights.Bold,
                    Foreground = Brushes.Black,
                    FontSize = 10
                };
                Canvas.SetLeft(label, node.Position.X - 5);
                Canvas.SetTop(label, node.Position.Y - 25);
                GraphCanvas.Children.Add(label);
            }

            // Highlight current node
            if (!string.IsNullOrEmpty(graphData.CurrentNode))
            {
                var currentNode = graphData.Nodes.FirstOrDefault(n => n.Id == graphData.CurrentNode);
                if (currentNode != null)
                {
                    var highlight = new Ellipse
                    {
                        Width = 24,
                        Height = 24,
                        Fill = Brushes.Red,
                        Opacity = 0.5
                    };
                    Canvas.SetLeft(highlight, currentNode.Position.X - 12);
                    Canvas.SetTop(highlight, currentNode.Position.Y - 12);
                    GraphCanvas.Children.Add(highlight);
                }
            }
        }

        protected override void OnClosed(EventArgs e)
        {
            _isRunning = false;
            _graphReceiverThread?.Join();
            base.OnClosed(e);
        }

        #region Data Classes
        public class GraphData
        {
            public List<Node> Nodes { get; set; } = new List<Node>();
            public List<Edge> Edges { get; set; } = new List<Edge>();
            public string CurrentNode { get; set; }
        }

        public class Node
        {
            public string Id { get; set; }
            public Point Position { get; set; }
        }

        public class Edge
        {
            public Point From { get; set; }
            public Point To { get; set; }
            public int Weight { get; set; }
        }

        public class PythonGraphData
        {
            public List<PythonNode> nodes { get; set; }
            public List<PythonEdge> edges { get; set; }
            public string current_node { get; set; }
        }

        public class PythonNode
        {
            public string id { get; set; }
            public double[] pos { get; set; }
        }

        public class PythonEdge
        {
            public string from { get; set; }
            public string to { get; set; }
            public int weight { get; set; }
        }
        #endregion
    }
}