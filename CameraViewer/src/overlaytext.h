#ifndef OVERLAYTEXT_H
#define OVERLAYTEXT_H

#include <QString>
#include <QColor>
#include <opencv2/core.hpp>

/**
 * @brief 在 OpenCV 图像上绘制 UTF-8（含中文）文本的工具。
 *
 * 背景：OpenCV 的 cv::putText 只支持 Hershey 矢量字体（ASCII），
 * 传入中文会全部渲染成 "?????"，这正是主界面门线标签
 * 「DOOR 上车↓」与离线检测框标签「ID:5 上车」出现问号的原因。
 *
 * 本工具借助 QImage + QPainter 渲染文本（QPainter 支持 Unicode 与 CJK 字体），
 * 再写回 BGR cv::Mat，从而正确显示中文。
 */
namespace OverlayText {

/// 在 BGR 图像 bgr 上绘制文本；origin 为文本基线左端（语义同 cv::putText 的 org）。
/// pixelSize 为字号（像素）；outline=true 时绘制黑色描边以提升可读性。
void draw(cv::Mat &bgr, const QString &text, const cv::Point &origin,
          const QColor &color, int pixelSize = 18, bool outline = true);

/// 测量文本包围盒（宽、高），用于绘制背景底板。
cv::Size measure(const QString &text, int pixelSize = 18);

} // namespace OverlayText

#endif // OVERLAYTEXT_H
