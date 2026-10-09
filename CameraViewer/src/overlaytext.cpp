#include "overlaytext.h"

#include <QImage>
#include <QPainter>
#include <QFont>
#include <QFontMetrics>
#include <QStringList>
#include <QRect>

#include <opencv2/imgproc.hpp>

namespace {

QFont makeFont(int pixelSize)
{
    QFont f;
    f.setPixelSize(qMax(1, pixelSize));
    f.setBold(true);
    f.setStyleStrategy(QFont::PreferAntialias);
#if QT_VERSION >= QT_VERSION_CHECK(5, 13, 0)
    // 优先选择系统内的中日韩字体，确保中文不再显示为方框/问号
    f.setFamilies(QStringList()
                  << QStringLiteral("Noto Sans CJK SC")
                  << QStringLiteral("Source Han Sans SC")
                  << QStringLiteral("WenQuanYi Micro Hei")
                  << QStringLiteral("WenQuanYi Zen Hei")
                  << QStringLiteral("Microsoft YaHei")
                  << QStringLiteral("Ubuntu")
                  << QStringLiteral("DejaVu Sans")
                  << QStringLiteral("Sans Serif"));
#else
    f.setFamily(QStringLiteral("Noto Sans CJK SC"));
#endif
    return f;
}

} // namespace

void OverlayText::draw(cv::Mat &bgr, const QString &text, const cv::Point &origin,
                       const QColor &color, int pixelSize, bool outline)
{
    if (bgr.empty() || bgr.type() != CV_8UC3 || text.isEmpty())
        return;

    // BGR -> RGB，用 QImage 包裹 rgb 的像素缓冲进行绘制
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
               QImage::Format_RGB888);

    {
        QPainter p(&img);
        p.setRenderHint(QPainter::Antialiasing, true);
        p.setRenderHint(QPainter::TextAntialiasing, true);
        p.setFont(makeFont(pixelSize));

        const QPoint pos(origin.x, origin.y);

        if (outline) {
            const QColor oc(0, 0, 0);
            p.setPen(oc);
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    if (dx == 0 && dy == 0) continue;
                    p.drawText(pos + QPoint(dx, dy), text);
                }
            }
        }

        p.setPen(color);
        p.drawText(pos, text);
    }

    // RGB -> BGR 写回原图（copyTo 尽量复用 bgr 既有内存）
    cv::Mat out;
    cv::cvtColor(rgb, out, cv::COLOR_RGB2BGR);
    out.copyTo(bgr);
}

cv::Size OverlayText::measure(const QString &text, int pixelSize)
{
    if (text.isEmpty()) return cv::Size(0, 0);
    QFontMetrics fm(makeFont(pixelSize));
    const QRect r = fm.boundingRect(text);
    return cv::Size(r.width(), r.height());
}
