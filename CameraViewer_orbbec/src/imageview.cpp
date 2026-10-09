#include "imageview.h"
#include "logger.h"

#include <QMouseEvent>
#include <QScrollBar>
#include <QWheelEvent>
#include <QPainter>
#include <QToolTip>

#include <cmath>
#include <cstring>

ImageView::ImageView(QWidget* parent)
    : QGraphicsView(parent)
{
    m_scene = new QGraphicsScene(this);
    m_item = new QGraphicsPixmapItem();
    m_scene->addItem(m_item);
    setScene(m_scene);

    setRenderHint(QPainter::Antialiasing, true);
    setRenderHint(QPainter::SmoothPixmapTransform, true);
    setDragMode(QGraphicsView::NoDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorViewCenter);
    setBackgroundBrush(QBrush(QColor(5, 10, 22)));
    setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    setCursor(Qt::ArrowCursor);
    setMouseTracking(true);   // 深度视图需要无按键即时显示悬停信息
}

void ImageView::setDepthData(int width, int height, const quint16 *dataMM)
{
    if (width <= 0 || height <= 0 || !dataMM) {
        m_depthEnabled = false;
        return;
    }
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    m_depthMM.resize(static_cast<int>(n));
    std::memcpy(m_depthMM.data(), dataMM, n * sizeof(quint16));
    m_depthW = width;
    m_depthH = height;
    m_depthEnabled = true;
}

void ImageView::setImage(const QImage& image)
{
    if (image.isNull()) {
        LOG_WARN("[ImageView] setImage received null image");
        return;
    }

    if (m_cachedPixmap.isNull() || m_cachedPixmap.size() != image.size()) {
        m_cachedPixmap = QPixmap::fromImage(image);
    } else {
        m_cachedPixmap.convertFromImage(image);
    }

    m_item->setPixmap(m_cachedPixmap);
    m_scene->setSceneRect(m_item->boundingRect());

    if (!m_hasImage) {
        m_hasImage = true;
        LOG_INFO(QString("[ImageView] First image set: %1x%2").arg(image.width()).arg(image.height()));
        fitImageInView();
    }
}

void ImageView::resetView()
{
    fitImageInView();
}

void ImageView::fitImageInView()
{
    if (!m_hasImage || m_item->pixmap().isNull()) return;
    resetTransform();
    fitInView(m_item, Qt::KeepAspectRatio);
    m_fitZoomFactor = transform().m11();
    m_zoomFactor = m_fitZoomFactor;
}

void ImageView::setDrawMode(DrawMode mode)
{
    m_drawMode = mode;
    m_drawingPoints.clear();
    m_rectDragging = false;
    m_draggingHandle = false;
    m_dragPolygonIndex = -1;
    m_dragVertexIndex = -1;
    if (mode == DrawMode::DrawPolygon || mode == DrawMode::DrawRectangle) {
        setCursor(Qt::CrossCursor);
    } else if (mode == DrawMode::EditRegion) {
        setCursor(Qt::ArrowCursor);
    } else {
        setCursor(Qt::ArrowCursor);
    }
    viewport()->update();
}

ImageView::DrawMode ImageView::drawMode() const
{
    return m_drawMode;
}

void ImageView::addRegionPolygon(const QVector<QPointF> &polygon)
{
    m_regionPolygons.append(polygon);
    viewport()->update();
}

void ImageView::clearRegionPolygons()
{
    m_regionPolygons.clear();
    m_doorLines.clear();
    m_drawingPoints.clear();
    viewport()->update();
}

void ImageView::setRegionPolygons(const QVector<QVector<QPointF>> &polygons)
{
    m_regionPolygons = polygons;
    viewport()->update();
}

void ImageView::setDoorLines(const QVector<QPair<QPointF, QPointF>> &lines)
{
    m_doorLines = lines;
    viewport()->update();
}

void ImageView::clearDoorLines()
{
    m_doorLines.clear();
    viewport()->update();
}

QVector<QPointF> ImageView::currentDrawingPoints() const
{
    return m_drawingPoints;
}

QVector<QVector<QPointF>> ImageView::regionPolygons() const
{
    return m_regionPolygons;
}

QSize ImageView::displayedImageSize() const
{
    if (m_cachedPixmap.isNull())
        return QSize();
    return m_cachedPixmap.size();
}

void ImageView::finishCurrentPolygon()
{
    if (m_drawingPoints.size() >= 3) {
        emit polygonCompleted(m_drawingPoints);
        m_drawingPoints.clear();
    } else {
        m_drawingPoints.clear();
    }
    viewport()->update();
}

void ImageView::finishCurrentRectangle()
{
    if (!m_rectDragging) return;
    m_rectDragging = false;

    // 将矩形的两个角点转换为4个顶点的多边形
    qreal x1 = qMin(m_rectStart.x(), m_rectEnd.x());
    qreal y1 = qMin(m_rectStart.y(), m_rectEnd.y());
    qreal x2 = qMax(m_rectStart.x(), m_rectEnd.x());
    qreal y2 = qMax(m_rectStart.y(), m_rectEnd.y());

    // 检查矩形大小是否合理
    if (qAbs(x2 - x1) < 5 || qAbs(y2 - y1) < 5) {
        LOG_WARN("[ImageView] Rectangle too small, ignored");
        viewport()->update();
        return;
    }

    QVector<QPointF> rectPoints;
    rectPoints.append(QPointF(x1, y1)); // 左上
    rectPoints.append(QPointF(x2, y1)); // 右上
    rectPoints.append(QPointF(x2, y2)); // 右下
    rectPoints.append(QPointF(x1, y2)); // 左下

    emit polygonCompleted(rectPoints);
    viewport()->update();
}

QPointF ImageView::mapToImage(const QPoint &viewPos) const
{
    QPointF scenePt = mapToScene(viewPos);
    return scenePt;
}

QPointF ImageView::mapFromImage(const QPointF &imgPt) const
{
    return mapFromScene(imgPt);
}

int ImageView::findHandleAt(const QPoint &viewPos, int &polygonIndex, int &vertexIndex) const
{
    polygonIndex = -1;
    vertexIndex = -1;
    double bestDist = HANDLE_HIT_SIZE + 1.0;

    for (int pi = 0; pi < m_regionPolygons.size(); ++pi) {
        const QVector<QPointF> &poly = m_regionPolygons[pi];
        for (int vi = 0; vi < poly.size(); ++vi) {
            QPointF vp = mapFromScene(poly[vi]);
            double dx = viewPos.x() - vp.x();
            double dy = viewPos.y() - vp.y();
            double dist = qSqrt(dx * dx + dy * dy);
            if (dist < bestDist) {
                bestDist = dist;
                polygonIndex = pi;
                vertexIndex = vi;
            }
        }
    }

    return (polygonIndex >= 0) ? 1 : 0;
}

void ImageView::updateCursorForPosition(const QPoint &viewPos)
{
    if (m_drawMode == DrawMode::EditRegion) {
        int pi, vi;
        if (findHandleAt(viewPos, pi, vi)) {
            setCursor(Qt::CrossCursor);
        } else {
            setCursor(Qt::ArrowCursor);
        }
    }
}

void ImageView::wheelEvent(QWheelEvent* event)
{
    if (!m_hasImage) return;
    if (event->angleDelta().y() == 0) return;

    const bool zoomIn = event->angleDelta().y() > 0;
    const double step = 1.15;
    const double nextZoom = zoomIn ? (m_zoomFactor * step) : (m_zoomFactor / step);
    const double minZoom = m_fitZoomFactor * 0.5;
    const double maxZoom = m_fitZoomFactor * 30.0;
    if (nextZoom < minZoom || nextZoom > maxZoom) return;

    const double ratio = nextZoom / m_zoomFactor;
    scale(ratio, ratio);
    m_zoomFactor = nextZoom;
    event->accept();
}

void ImageView::mousePressEvent(QMouseEvent* event)
{
    // EditRegion 模式：拖拽角点
    if (m_drawMode == DrawMode::EditRegion) {
        if (event->button() == Qt::LeftButton) {
            int pi, vi;
            if (findHandleAt(event->pos(), pi, vi)) {
                m_draggingHandle = true;
                m_dragPolygonIndex = pi;
                m_dragVertexIndex = vi;
                event->accept();
                return;
            }
            // 没点到手柄，则平移
            m_panning = true;
            m_lastMousePos = event->pos();
            setCursor(Qt::ClosedHandCursor);
            event->accept();
            return;
        }
    }

    if (m_drawMode == DrawMode::DrawPolygon) {
        if (event->button() == Qt::LeftButton) {
            QPointF imgPt = mapToImage(event->pos());
            // 允许在有图像或无图像时都能绘制
            m_drawingPoints.append(imgPt);
            emit drawingPointAdded(m_drawingPoints.size());
            viewport()->update();
            event->accept();
            return;
        } else if (event->button() == Qt::RightButton) {
            finishCurrentPolygon();
            event->accept();
            return;
        }
    }

    // DrawRectangle 模式：按下左键开始拖拽
    if (m_drawMode == DrawMode::DrawRectangle) {
        if (event->button() == Qt::LeftButton) {
            m_rectStart = mapToImage(event->pos());
            m_rectEnd = m_rectStart;
            m_rectDragging = true;
            viewport()->update();
            event->accept();
            return;
        } else if (event->button() == Qt::RightButton) {
            // 右键取消
            m_rectDragging = false;
            viewport()->update();
            event->accept();
            return;
        }
    }

    // View 模式：保持原有的拖拽行为
    if (event->button() == Qt::LeftButton) {
        m_panning = true;
        m_lastMousePos = event->pos();
        setCursor(Qt::ClosedHandCursor);
    }
    QGraphicsView::mousePressEvent(event);
}

void ImageView::mouseMoveEvent(QMouseEvent* event)
{
    // 深度视图（只读）：鼠标悬停显示 像素坐标 + 距离/高度(m)
    if (m_depthEnabled && m_drawMode == DrawMode::View && m_hasImage) {
        const QPointF imgPt = mapToImage(event->pos());
        const int ix = qRound(imgPt.x());
        const int iy = qRound(imgPt.y());
        if (ix >= 0 && ix < m_depthW && iy >= 0 && iy < m_depthH) {
            const quint16 d = m_depthMM[iy * m_depthW + ix];
            const QString txt = (d == 0)
                ? QString("像素坐标 (%1, %2)   深度: 无有效值(0)").arg(ix).arg(iy)
                : QString("像素坐标 (%1, %2)   距离/高度: %3 m")
                      .arg(ix).arg(iy).arg(d / 1000.0, 0, 'f', 3);
            QToolTip::showText(event->globalPos(), txt, this);
        }
    }

    // EditRegion 模式：拖拽角点
    if (m_drawMode == DrawMode::EditRegion) {
        if (m_draggingHandle && m_dragPolygonIndex >= 0 && m_dragVertexIndex >= 0) {
            QPointF imgPt = mapToImage(event->pos());
            // 限制在图像范围内
            if (m_hasImage) {
                imgPt.setX(qBound(0.0, imgPt.x(), (double)m_cachedPixmap.width()));
                imgPt.setY(qBound(0.0, imgPt.y(), (double)m_cachedPixmap.height()));
            }
            if (m_dragPolygonIndex < m_regionPolygons.size() &&
                m_dragVertexIndex < m_regionPolygons[m_dragPolygonIndex].size()) {
                m_regionPolygons[m_dragPolygonIndex][m_dragVertexIndex] = imgPt;
                viewport()->update();
            }
            event->accept();
            return;
        }
        if (m_panning) {
            const QPoint delta = event->pos() - m_lastMousePos;
            m_lastMousePos = event->pos();
            horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
            verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
            event->accept();
            return;
        }
        // 更新光标
        updateCursorForPosition(event->pos());
        m_mousePos = event->pos();
        viewport()->update();
        event->accept();
        return;
    }

    if (m_drawMode == DrawMode::DrawPolygon) {
        m_mousePos = event->pos();
        viewport()->update();
        event->accept();
        return;
    }

    // DrawRectangle 模式：拖拽更新矩形终点
    if (m_drawMode == DrawMode::DrawRectangle) {
        if (m_rectDragging) {
            m_rectEnd = mapToImage(event->pos());
            m_mousePos = event->pos();
            viewport()->update();
        }
        event->accept();
        return;
    }

    if (m_panning) {
        const QPoint delta = event->pos() - m_lastMousePos;
        m_lastMousePos = event->pos();
        horizontalScrollBar()->setValue(horizontalScrollBar()->value() - delta.x());
        verticalScrollBar()->setValue(verticalScrollBar()->value() - delta.y());
    }
    QGraphicsView::mouseMoveEvent(event);
}

void ImageView::mouseReleaseEvent(QMouseEvent* event)
{
    if (m_drawMode == DrawMode::EditRegion) {
        if (m_draggingHandle && event->button() == Qt::LeftButton) {
            m_draggingHandle = false;
            // 发送多边形修改信号
            if (m_dragPolygonIndex >= 0 && m_dragPolygonIndex < m_regionPolygons.size()) {
                emit polygonModified(m_dragPolygonIndex, m_regionPolygons[m_dragPolygonIndex]);
            }
            m_dragPolygonIndex = -1;
            m_dragVertexIndex = -1;
            event->accept();
            return;
        }
        if (m_panning && event->button() == Qt::LeftButton) {
            m_panning = false;
            setCursor(Qt::ArrowCursor);
            event->accept();
            return;
        }
        event->accept();
        return;
    }

    if (m_drawMode == DrawMode::DrawPolygon) {
        event->accept();
        return;
    }

    // DrawRectangle 模式：释放左键完成矩形
    if (m_drawMode == DrawMode::DrawRectangle) {
        if (event->button() == Qt::LeftButton && m_rectDragging) {
            m_rectEnd = mapToImage(event->pos());
            finishCurrentRectangle();
        }
        event->accept();
        return;
    }

    if (event->button() == Qt::LeftButton) {
        m_panning = false;
        if (m_drawMode == DrawMode::View)
            setCursor(Qt::ArrowCursor);
    }
    QGraphicsView::mouseReleaseEvent(event);
}

void ImageView::resizeEvent(QResizeEvent* event)
{
    QGraphicsView::resizeEvent(event);
    Q_UNUSED(event);
}

void ImageView::mouseDoubleClickEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton && m_hasImage) {
        if (m_drawMode == DrawMode::View) {
            fitImageInView();
            event->accept();
            return;
        }
    }
    QGraphicsView::mouseDoubleClickEvent(event);
}

void ImageView::paintEvent(QPaintEvent* event)
{
    // 先绘制默认的场景内容
    QGraphicsView::paintEvent(event);

    // 然后用 QPainter 在 viewport 上绘制多边形叠加
    // DrawRectangle 模式下即使 m_hasImage=false 也允许绘制
    bool hasDrawContent = !m_regionPolygons.isEmpty() || !m_doorLines.isEmpty()
                          || !m_drawingPoints.isEmpty() || m_rectDragging;
    if (m_drawMode != DrawMode::DrawRectangle && (!m_hasImage || !hasDrawContent))
        return;
    if (m_drawMode == DrawMode::DrawRectangle && !hasDrawContent && m_regionPolygons.isEmpty())
        return;

    QPainter painter(viewport());
    painter.setRenderHint(QPainter::Antialiasing, true);

    // ===== 绘制已确认的区域多边形 =====
    for (int i = 0; i < m_regionPolygons.size(); ++i) {
        const QVector<QPointF> &poly = m_regionPolygons[i];
        if (poly.size() < 3) continue;

        // 转换图像坐标到视图坐标
        QPolygonF viewPoly;
        for (const QPointF &imgPt : poly) {
            viewPoly.append(mapFromScene(imgPt));
        }

        // 半透明蓝色填充 (rgba: 0, 100, 255, 50)
        QColor fillColor(0, 100, 255, 50);
        painter.setBrush(QBrush(fillColor));
        // 绿色边框 (2px宽度)
        QPen borderPen(QColor(0, 255, 0), 2, Qt::SolidLine);
        painter.setPen(borderPen);
        painter.drawPolygon(viewPoly);

        // 绘制角点拖拽手柄 (8x8白色方块 + 1px绿色边框)
        for (const QPointF &vp : viewPoly) {
            // 8x8 白色小方块，居中于顶点
            QRectF handleRect(vp.x() - HANDLE_SIZE, vp.y() - HANDLE_SIZE,
                              HANDLE_SIZE * 2, HANDLE_SIZE * 2);
            painter.setBrush(QBrush(Qt::white));
            painter.setPen(QPen(QColor(0, 255, 0), 1));
            painter.drawRect(handleRect);
        }

        // EditRegion 模式下高亮悬停的手柄
        if (m_drawMode == DrawMode::EditRegion) {
            int pi, vi;
            if (findHandleAt(m_mousePos, pi, vi) && pi == i) {
                QPointF hoverPt = viewPoly[vi];
                QRectF hoverRect(hoverPt.x() - HANDLE_SIZE - 2, hoverPt.y() - HANDLE_SIZE - 2,
                                 HANDLE_SIZE * 2 + 4, HANDLE_SIZE * 2 + 4);
                painter.setBrush(Qt::NoBrush);
                painter.setPen(QPen(QColor(255, 255, 0), 2, Qt::DashLine));
                painter.drawRect(hoverRect);
            }
        }
    }

    // ===== 绘制虚拟门线（Door Line）=====
    for (const auto &dl : m_doorLines) {
        const QPointF a = mapFromScene(dl.first);
        const QPointF b = mapFromScene(dl.second);

        // 外发光（较粗的半透明底线）→ 主体亮青色实线
        painter.setBrush(Qt::NoBrush);
        painter.setPen(QPen(QColor(0, 120, 180, 120), 7, Qt::SolidLine, Qt::RoundCap));
        painter.drawLine(a, b);
        painter.setPen(QPen(QColor(0, 220, 255), 3, Qt::SolidLine, Qt::RoundCap));
        painter.drawLine(a, b);

        // 两端点标记
        painter.setBrush(QBrush(QColor(0, 220, 255)));
        painter.setPen(QPen(QColor(255, 255, 255), 1));
        painter.drawEllipse(a, 5, 5);
        painter.drawEllipse(b, 5, 5);

        // 门线中点标注（垂直方向刻度，示意“跨越门线”）
        const QPointF mid((a.x() + b.x()) * 0.5, (a.y() + b.y()) * 0.5);
        QPointF dir = b - a;
        const double len = std::sqrt(dir.x() * dir.x() + dir.y() * dir.y());
        if (len > 1e-6) {
            dir /= len;
            const QPointF nrm(-dir.y(), dir.x());
            painter.setPen(QPen(QColor(0, 220, 255), 2));
            painter.drawLine(mid - nrm * 10.0, mid + nrm * 10.0);
        }
    }

    // ===== 绘制当前正在绘制的多边形 =====
    if (!m_drawingPoints.isEmpty()) {
        QColor drawColor(255, 255, 0); // 黄色表示正在绘制
        QPolygonF viewPoly;
        for (const QPointF &imgPt : m_drawingPoints) {
            viewPoly.append(mapFromScene(imgPt));
        }

        // 画已有的线段
        painter.setPen(QPen(drawColor, 2, Qt::DashLine));
        painter.setBrush(Qt::NoBrush);
        if (viewPoly.size() > 1) {
            painter.drawPolyline(viewPoly);
        }

        // 画到鼠标位置的预览线
        if (viewPoly.size() > 0) {
            QPen previewPen(drawColor, 1, Qt::DotLine);
            painter.setPen(previewPen);
            painter.drawLine(viewPoly.last(), QPointF(m_mousePos));
            // 闭合预览线
            painter.drawLine(QPointF(m_mousePos), viewPoly.first());
        }

        // 绘制顶点
        painter.setBrush(QBrush(drawColor));
        painter.setPen(QPen(drawColor, 1));
        for (const QPointF &vp : viewPoly) {
            painter.drawEllipse(vp, 5, 5);
        }
    }

    // ===== 绘制矩形拖拽预览 =====
    if (m_rectDragging && m_drawMode == DrawMode::DrawRectangle) {
        QPointF topLeft = mapFromScene(qMin(m_rectStart.x(), m_rectEnd.x()),
                                       qMin(m_rectStart.y(), m_rectEnd.y()));
        QPointF bottomRight = mapFromScene(qMax(m_rectStart.x(), m_rectEnd.x()),
                                           qMax(m_rectStart.y(), m_rectEnd.y()));
        QRectF previewRect(topLeft, bottomRight);

        // 半透明黄色填充
        QColor fillColor(255, 255, 0, 40);
        painter.setBrush(QBrush(fillColor));
        painter.setPen(QPen(QColor(255, 255, 0), 2, Qt::DashLine));
        painter.drawRect(previewRect);

        // 显示尺寸文本
        int w = static_cast<int>(qAbs(m_rectEnd.x() - m_rectStart.x()));
        int h = static_cast<int>(qAbs(m_rectEnd.y() - m_rectStart.y()));
        QString sizeText = QString("%1×%2").arg(w).arg(h);
        painter.setPen(QPen(QColor(255, 255, 200), 1));
        painter.setFont(QFont("Consolas", 10));
        painter.drawText(previewRect.center() - QPointF(20, 0), sizeText);
    }

    painter.end();
}
