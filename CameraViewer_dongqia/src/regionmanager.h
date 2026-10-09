#ifndef REGIONMANAGER_H
#define REGIONMANAGER_H

#include <QObject>
#include <QVector>
#include <QPointF>
#include <QString>
#include <QColor>

// 进入方向枚举
enum class EntryDirection {
    TopToBottom,    // 从上往下进入
    BottomToTop     // 从下往上进入
};

struct DetectionRegion {
    int id;
    QVector<QPointF> polygon;  // 多边形顶点（归一化坐标 0~1）
    QString name;
    QColor color;
    EntryDirection entryDirection = EntryDirection::TopToBottom; // 默认从上往下
};

class RegionManager : public QObject {
    Q_OBJECT
public:
    explicit RegionManager(QObject *parent = nullptr);

    void addRegion(const DetectionRegion &region);
    void removeRegion(int id);
    void clearRegions();
    QVector<DetectionRegion> regions() const;

    bool isPointInRegion(const QPointF &normalizedPoint, int regionId) const;
    bool isPointInAnyRegion(const QPointF &normalizedPoint) const;
    int pointInWhichRegion(const QPointF &normalizedPoint) const;

    void saveRegions(const QString &path);
    void loadRegions(const QString &path);

signals:
    void regionsChanged();

private:
    QVector<DetectionRegion> m_regions;
    int m_nextId;

    bool pointInPolygon(const QPointF &point, const QVector<QPointF> &polygon) const;
};

#endif // REGIONMANAGER_H
