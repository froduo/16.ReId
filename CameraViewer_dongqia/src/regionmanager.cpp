#include "regionmanager.h"
#include "logger.h"

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

RegionManager::RegionManager(QObject *parent)
    : QObject(parent)
    , m_nextId(1)
{
}

void RegionManager::addRegion(const DetectionRegion &region)
{
    DetectionRegion r = region;
    if (r.id <= 0)
        r.id = m_nextId++;
    else if (r.id >= m_nextId)
        m_nextId = r.id + 1;

    m_regions.append(r);
    emit regionsChanged();
}

void RegionManager::removeRegion(int id)
{
    for (int i = 0; i < m_regions.size(); ++i) {
        if (m_regions[i].id == id) {
            m_regions.removeAt(i);
            emit regionsChanged();
            return;
        }
    }
}

void RegionManager::clearRegions()
{
    if (!m_regions.isEmpty()) {
        m_regions.clear();
        emit regionsChanged();
    }
}

QVector<DetectionRegion> RegionManager::regions() const
{
    return m_regions;
}

bool RegionManager::pointInPolygon(const QPointF &point, const QVector<QPointF> &polygon) const
{
    // Ray-casting algorithm
    int n = polygon.size();
    if (n < 3) return false;

    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const QPointF &pi = polygon[i];
        const QPointF &pj = polygon[j];

        if (((pi.y() > point.y()) != (pj.y() > point.y())) &&
            (point.x() < (pj.x() - pi.x()) * (point.y() - pi.y()) / (pj.y() - pi.y()) + pi.x())) {
            inside = !inside;
        }
    }
    return inside;
}

bool RegionManager::isPointInRegion(const QPointF &normalizedPoint, int regionId) const
{
    for (const auto &r : m_regions) {
        if (r.id == regionId) {
            return pointInPolygon(normalizedPoint, r.polygon);
        }
    }
    return false;
}

bool RegionManager::isPointInAnyRegion(const QPointF &normalizedPoint) const
{
    for (const auto &r : m_regions) {
        if (pointInPolygon(normalizedPoint, r.polygon))
            return true;
    }
    return false;
}

int RegionManager::pointInWhichRegion(const QPointF &normalizedPoint) const
{
    for (const auto &r : m_regions) {
        if (pointInPolygon(normalizedPoint, r.polygon))
            return r.id;
    }
    return -1;
}

void RegionManager::saveRegions(const QString &path)
{
    QJsonArray regionsArray;
    for (const auto &r : m_regions) {
        QJsonObject regionObj;
        regionObj["id"] = r.id;
        regionObj["name"] = r.name;
        regionObj["color"] = r.color.name();
        regionObj["entryDirection"] = (r.entryDirection == EntryDirection::TopToBottom) ? "TopToBottom" : "BottomToTop";

        QJsonArray polyArray;
        for (const auto &pt : r.polygon) {
            QJsonObject ptObj;
            ptObj["x"] = pt.x();
            ptObj["y"] = pt.y();
            polyArray.append(ptObj);
        }
        regionObj["polygon"] = polyArray;
        regionsArray.append(regionObj);
    }

    QJsonDocument doc(regionsArray);
    QFile file(path);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(doc.toJson());
        file.close();
        LOG_INFO(QString("[RegionManager] Saved %1 regions to %2").arg(m_regions.size()).arg(path));
    } else {
        LOG_ERR("[RegionManager] Failed to save regions to " + path);
    }
}

void RegionManager::loadRegions(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        LOG_WARN("[RegionManager] No region file found: " + path);
        return;
    }

    QByteArray data = file.readAll();
    file.close();

    QJsonDocument doc = QJsonDocument::fromJson(data);
    if (!doc.isArray()) {
        LOG_ERR("[RegionManager] Invalid region file format: " + path);
        return;
    }

    m_regions.clear();
    QJsonArray regionsArray = doc.array();
    for (const auto &val : regionsArray) {
        QJsonObject obj = val.toObject();
        DetectionRegion r;
        r.id = obj["id"].toInt();
        r.name = obj["name"].toString();
        r.color = QColor(obj["color"].toString());
        QString dirStr = obj["entryDirection"].toString("TopToBottom");
        r.entryDirection = (dirStr == "BottomToTop") ? EntryDirection::BottomToTop : EntryDirection::TopToBottom;

        QJsonArray polyArray = obj["polygon"].toArray();
        for (const auto &ptVal : polyArray) {
            QJsonObject ptObj = ptVal.toObject();
            r.polygon.append(QPointF(ptObj["x"].toDouble(), ptObj["y"].toDouble()));
        }

        if (r.id >= m_nextId)
            m_nextId = r.id + 1;

        m_regions.append(r);
    }

    LOG_INFO(QString("[RegionManager] Loaded %1 regions from %2").arg(m_regions.size()).arg(path));
    emit regionsChanged();
}
