using System;
using System.Globalization;
using System.Windows.Data;
using System.Windows.Media;

namespace Controller.Common
{
    public class HeadingToColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            // مقدار ورودی، جهت فعلی ربات است
            string? currentHeading = value as string;
            // پارامتر، جهتی است که این فلش نمایش می‌دهد
            string? arrowDirection = parameter as string;

            if (currentHeading == arrowDirection)
            {
                // رنگ فعال (سبز روشن)
                return new SolidColorBrush(Colors.LawnGreen);
            }
            // رنگ غیرفعال (خاکستری)
            return new SolidColorBrush(Colors.Gray);
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}