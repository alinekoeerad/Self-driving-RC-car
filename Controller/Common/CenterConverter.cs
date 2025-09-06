using System;
using System.Globalization;
using System.Windows.Data;

// Make sure the namespace is exactly this
namespace Controller.Common
{
    // Make sure the class is declared as "public"
    public class CenterConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is double coordinate && parameter is string sizeStr && double.TryParse(sizeStr, out double size))
            {
                return coordinate - (size / 2);
            }
            return value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            // This is not needed for one-way binding.
            throw new NotImplementedException();
        }
    }
}