#ifndef IMAGEVIEW_H
#define IMAGEVIEW_H

#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QImage>
#include <QPixmap>
#include <QVector>
#include <QPointF>
#include <QColor>
#include <QPair>

class ImageView : public QGraphicsView
{
    Q_OBJECT

public:
    enum class DrawMode { View, DrawPolygon, DrawRectangle, EditRegion };

    explicit ImageView(QWidget* parent = nullptr);
    void setImage(const QImage& image);
    void resetView();

    // 绘图模式
    void setDrawMode(DrawMode mode);
    DrawMode drawMode() const;

    // 区域多边形管理
    void addRegionPolygon(const QVector<QPointF> &polygon);
    void clearRegionPolygons();
    void setRegionPolygons(const QVector<QVector<QPointF>> &polygons);
    QVector<QPointF> currentDrawingPoints() const;
    void finishCurrentPolygon();
    void finishCurrentRectangle(); // 完成当前矩形绘制

    // ===== 深度数据（供鼠标悬停查询像素坐标 → 距离/高度）=====
    /// 传入与显示伪彩图同分辨率的深度（CV_16U，毫米；0 表示无效）
    void setDepthData(int width, int height, const quint16 *dataMM);
    void clearDepthData() { m_depthEnabled = false; m_depthMM.clear(); }

    // ===== 虚拟门线管理（图像归一化/图像坐标，两端点）=====
    void setDoorLines(const QVector<QPair<QPointF, QPointF>> &lines);
    void clearDoorLines();
    QVector<QPair<QPointF, QPointF>> doorLines() const { return m_doorLines; }

    // 获取多边形数据（用于外部同步）
    QVector<QVector<QPointF>> regionPolygons() const;

    // 获取当前显示图像尺寸（像素）
    QSize displayedImageSize() const;

signals:
    void polygonCompleted(const QVector<QPointF> &points);
    void polygonModified(int index, const QVector<QPointF> &points);
    void drawingPointAdded(int totalPoints); // 绘制模式下新增顶点时发出

protected:
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void paintEvent(QPaintEvent* event) override;

private:
    void fitImageInView();
    QPointF mapToImage(const QPoint &viewPos) const;
    QPointF mapFromImage(const QPointF &imgPt) const;
    int findHandleAt(const QPoint &viewPos, int &polygonIndex, int &vertexIndex) const;
    void updateCursorForPosition(const QPoint &viewPos);

private:
    QGraphicsScene* m_scene {nullptr};
    QGraphicsPixmapItem* m_item {nullptr};
    bool m_panning {false};
    QPoint m_lastMousePos;
    bool m_hasImage {false};
    double m_zoomFactor {1.0};
    double m_fitZoomFactor {1.0};
    QPixmap m_cachedPixmap;

    // 绘图模式
    DrawMode m_drawMode = DrawMode::View;
    QVector<QPointF> m_drawingPoints;              // 当前正在绘制的点（图像坐标）
    QVector<QVector<QPointF>> m_regionPolygons;    // 所有已绘制区域（图像坐标）
    QVector<QPair<QPointF, QPointF>> m_doorLines;  // 虚拟门线（图像坐标，两端点）
    QColor m_currentDrawColor = QColor(0, 255, 0);
    QPoint m_mousePos;                             // 鼠标当前位置（用于实时预览）

    // 深度悬停查询
    bool m_depthEnabled {false};
    int m_depthW {0};
    int m_depthH {0};
    QVector<quint16> m_depthMM;                    // 深度原始值（毫米）

    // 矩形绘制
    bool m_rectDragging {false};                   // 是否正在拖拽矩形
    QPointF m_rectStart;                           // 矩形起始点（图像坐标）
    QPointF m_rectEnd;                             // 矩形结束点（图像坐标）

    // 角点拖拽
    bool m_draggingHandle {false};
    int m_dragPolygonIndex {-1};
    int m_dragVertexIndex {-1};
    static const int HANDLE_SIZE = 4;       // 手柄半径（视图像素）
    static const int HANDLE_HIT_SIZE = 10;  // 悬停检测范围（视图像素）
};

#endif // IMAGEVIEW_H
