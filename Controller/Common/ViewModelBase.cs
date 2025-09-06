using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace Controller.Common
{
    // A base class for ViewModels to handle property change notifications.
    public abstract class ViewModelBase : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;

        protected void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}