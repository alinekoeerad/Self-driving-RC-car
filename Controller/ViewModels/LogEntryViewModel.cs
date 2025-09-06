using Controller.Common;
using System.Windows.Media;

namespace Controller.ViewModels
{
    // Represents a single log entry in the UI.
    public class LogEntryViewModel : ViewModelBase
    {
        public string Timestamp { get; }
        public string Message { get; }
        public Brush MessageColor { get; }

        public LogEntryViewModel(string message)
        {
            Timestamp = System.DateTime.Now.ToString("HH:mm:ss");
            Message = message;

            // Simple color coding based on message content
            if (message.StartsWith("❌") || message.ToLower().Contains("error"))
                MessageColor = Brushes.IndianRed;
            else if (message.StartsWith("✅") || message.ToLower().Contains("success"))
                MessageColor = Brushes.LightGreen;
            else if (message.StartsWith("⚠️") || message.ToLower().Contains("warning"))
                MessageColor = Brushes.Orange;
            else
                MessageColor = Brushes.LightGray;
        }
    }
}