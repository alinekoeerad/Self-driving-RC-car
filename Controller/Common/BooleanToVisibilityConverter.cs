using System;
using System.Globalization;
using System.Windows;
using System.Windows.Data;

namespace Controller.Common
{
    // A simple converter that converts a boolean to a Visibility value.
    public class BooleanToVisibilityConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            bool param = parameter as string == "invert";
            bool val = value is bool b && b;

            if (param) val = !val;

            return val ? Visibility.Visible : Visibility.Collapsed;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}