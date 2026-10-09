#ifndef REGIONMANAGER_H
#define REGIONMANAGER_H

#include <QObject>
#include <QVector>
#include <QPointF>
#include <QString>
#include <QColor>

// 进入方向枚举
enum class EntryDirection {
    TopToBottom,    // 从上往下进入（= 上车）
    BottomToTop     // 从下往上进入（= 下车）
};

// 区域几何形态
enum class RegionShape {
    Polygon,        // 多边形区域（旧行为：人必须是“完整穿越整个区域”）
    DoorLine        // 虚拟门线（新行为：只需跨越一条线即可判定上/下车）
};

struct DetectionRegion {
    int id;
    // ===== 形态 =====
    RegionShape shape = RegionShape::Polygon;
    // ===== 多边形（shape == Polygon）=====
    QVector<QPointF> polygon;  // 多边形顶点（归一化坐标 0~1）
    // ===== 虚拟门线（shape == DoorLine，归一化坐标 0~1）=====
    QPointF lineStart;         // 门线起点
    QPointF lineEnd;           // 门线终点
    QString name;
    QColor color;
    EntryDirection entryDirection = EntryDirection::TopToBottom; // 默认从上往下(上车)

    /// 是否显式配置了一条有效门线
    bool hasExplicitLine() const {
        return shape == RegionShape::DoorLine && lineStart != lineEnd;
    }
};

class RegionManager : public QObject {
    Q_OBJECT
public:
    explicit RegionManager(QObject *parent = nullptr);

    void addRegion(const DetectionRegion &region);
    void removeRegion(int id);
    void clearRegions();
    /// 返回区域列表（const 引用：避免返回临时量被误绑定引用导致悬垂）
    const QVector<DetectionRegion> &regions() const;

    bool isPointInRegion(const QPointF &normalizedPoint, int regionId) const;
    bool isPointInAnyRegion(const QPointF &normalizedPoint) const;
    int pointInWhichRegion(const QPointF &normalizedPoint) const;

    void saveRegions(const QString &path);
    void loadRegions(const QString &path);

    /**
     * @brief 计算某个「检测区域」对应的虚拟门线（归一化坐标）
     *
     * 规则：
     *   - 若区域显式定义了门线（shape==DoorLine 且两端有效），直接返回该门线；
     *   - 否则由多边形外接矩形派生一条水平门线：
     *       门线位于外接矩形「纵向中线」，横跨左、右边界。
     *     这样旧工程里已绘制的通道区域可**零配置**升级为“门线检测”。
     *
     * @param region 目标区域
     * @param a      输出：门线起点
     * @param b      输出：门线终点
     * @return 是否成功得到有效门线
     */
    static bool regionDoorLine(const DetectionRegion &region, QPointF &a, QPointF &b);

signals:
    void regionsChanged();

private:
    QVector<DetectionRegion> m_regions;
    int m_nextId;

    bool pointInPolygon(const QPointF &point, const QVector<QPointF> &polygon) const;
};

#endif // REGIONMANAGER_H
