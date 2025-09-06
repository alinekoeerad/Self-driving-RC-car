using System.Collections.Generic;
using Newtonsoft.Json;

namespace Controller.Models
{
    /// <summary>
    /// Data Transfer Object (DTO) to match the JSON packet from the Python server.
    /// </summary>
    public class PythonData
    {
        [JsonProperty("log")]
        public string Log { get; set; } = string.Empty;

        [JsonProperty("current_node")]
        public string? CurrentNode { get; set; }

        [JsonProperty("image")]
        public string? Image { get; set; }

        [JsonProperty("is_connected")]
        public bool IsConnected { get; set; }

        // Property to receive the calculated path for highlighting
        [JsonProperty("navigation_path")]
        public List<string> NavigationPath { get; set; } = new List<string>();

        [JsonProperty("nodes")]
        public List<PythonNode> Nodes { get; set; } = new List<PythonNode>();

        [JsonProperty("edges")]
        public List<PythonEdge> Edges { get; set; } = new List<PythonEdge>();

        // ADDED: Property to receive the robot's current orientation
        [JsonProperty("robot_heading")]
        public string RobotHeading { get; set; } = string.Empty;
    }

    public class PythonNode
    {
        [JsonProperty("id")]
        public string Id { get; set; } = string.Empty;

        [JsonProperty("pos")]
        public List<double> Position { get; set; } = new List<double>();
    }

    public class PythonEdge
    {
        [JsonProperty("from")]
        public string From { get; set; } = string.Empty;

        [JsonProperty("to")]
        public string To { get; set; } = string.Empty;
    }
}