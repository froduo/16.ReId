#include "tripdatastore.h"
#include "logger.h"

#include <QDir>
#include <QFileInfo>
#include <QStringList>
#include <QSqlDatabase>
#include <QSqlQuery>
#include <QSqlError>

#include <algorithm>

// ===================== 抽象基类 =====================

TripDataStore::~TripDataStore() = default;

QString TripDataStore::stopKey(const QString &tripId, const QString &stationId)
{
    return tripId + QChar(0x1f) + stationId;
}

TripDataStore *TripDataStore::create(const QString &backend)
{
    const QString b = backend.trimmed().toLower();
    if (b == QLatin1String("postgresql") || b == QLatin1String("postgres")
        || b == QLatin1String("pg") || b == QLatin1String("psql")) {
        return new SqlTripStore(QStringLiteral("QPSQL"), QStringLiteral("tripstore_pg"));
    }
    // 默认 SQLite
    return new SqlTripStore(QStringLiteral("QSQLITE"), QStringLiteral("tripstore_sqlite"));
}

// ===================== 通用 SQL 实现 =====================

SqlTripStore::SqlTripStore(const QString &driver, const QString &connectionName)
    : m_driver(driver)
    , m_connName(connectionName)
{
    m_backend = (driver == QLatin1String("QPSQL")) ? QStringLiteral("postgresql")
                                                   : QStringLiteral("sqlite");
}

SqlTripStore::~SqlTripStore()
{
    close();
}

bool SqlTripStore::open(const QVariantMap &options, QString *error)
{
    if (m_open) return true;

    if (!QSqlDatabase::isDriverAvailable(m_driver)) {
        const QString msg = QStringLiteral("数据库驱动不可用: %1（PostgreSQL 需 Qt QPSQL 插件，"
                                           "SQLite 需 QSQLITE 插件）").arg(m_driver);
        if (error) *error = msg;
        LOG_ERR("[TripDataStore] " + msg);
        return false;
    }

    QSqlDatabase db = QSqlDatabase::contains(m_connName)
            ? QSqlDatabase::database(m_connName)
            : QSqlDatabase::addDatabase(m_driver, m_connName);

    if (m_backend == QLatin1String("sqlite")) {
        QString path = options.value(QStringLiteral("path")).toString();
        if (path.isEmpty())
            path = QDir::current().absoluteFilePath(QStringLiteral("trip_data/bus_trips.db"));
        // 相对路径按程序目录解析
        if (QFileInfo(path).isRelative())
            path = QDir::current().absoluteFilePath(path);
        QDir().mkpath(QFileInfo(path).absolutePath());
        db.setDatabaseName(path);
    } else {
        db.setHostName(options.value(QStringLiteral("host"), QStringLiteral("127.0.0.1")).toString());
        db.setPort(options.value(QStringLiteral("port"), 5432).toInt());
        db.setDatabaseName(options.value(QStringLiteral("dbname"), QStringLiteral("bus")).toString());
        db.setUserName(options.value(QStringLiteral("user"), QStringLiteral("postgres")).toString());
        db.setPassword(options.value(QStringLiteral("password")).toString());
        const QString opts = options.value(QStringLiteral("connectOptions")).toString();
        if (!opts.isEmpty()) db.setConnectOptions(opts);
    }

    if (!db.open()) {
        const QString msg = QStringLiteral("打开 %1 失败: %2")
                                .arg(m_backend, db.lastError().text());
        if (error) *error = msg;
        LOG_ERR("[TripDataStore] " + msg);
        return false;
    }

    m_open = true;

    if (!createTables(error)) {
        close();
        return false;
    }

    LOG_INFO(QString("[TripDataStore] %1 后端已就绪, 连接=%2, 位置=%3")
                 .arg(m_backend, m_connName, db.databaseName()));
    return true;
}

void SqlTripStore::close()
{
    if (QSqlDatabase::contains(m_connName)) {
        {
            QSqlDatabase db = QSqlDatabase::database(m_connName);
            if (db.isOpen()) db.close();
        }
        QSqlDatabase::removeDatabase(m_connName);
    }
    m_open = false;
}

bool SqlTripStore::createTables(QString *error)
{
    QSqlDatabase db = QSqlDatabase::database(m_connName);
    const bool pg = (m_backend == QLatin1String("postgresql"));

    const QString seqType = pg ? QStringLiteral("BIGSERIAL PRIMARY KEY")
                               : QStringLiteral("INTEGER PRIMARY KEY AUTOINCREMENT");

    const QStringList ddl = {
        QStringLiteral("CREATE TABLE IF NOT EXISTS trips ("
                       "trip_id TEXT PRIMARY KEY, vehicle_id TEXT, "
                       "started_at TEXT, ended_at TEXT)"),
        QStringLiteral("CREATE TABLE IF NOT EXISTS stop_stats ("
                       "trip_id TEXT, station_id TEXT, station_name TEXT, "
                       "boarding INTEGER DEFAULT 0, alighting INTEGER DEFAULT 0, "
                       "onboard INTEGER DEFAULT 0, updated_at TEXT, "
                       "PRIMARY KEY (trip_id, station_id))"),
        QStringLiteral("CREATE TABLE IF NOT EXISTS passenger_events ("
                       "seq %1, trip_id TEXT, station_id TEXT, station_name TEXT, "
                       "passenger_id TEXT, event_type TEXT, recorded_at TEXT)")
            .arg(seqType)
    };

    for (const QString &sql : ddl) {
        QSqlQuery q(db);
        if (!q.exec(sql)) {
            const QString msg = QStringLiteral("建表失败: %1 | SQL=%2").arg(q.lastError().text(), sql);
            if (error) *error = msg;
            LOG_ERR("[TripDataStore] " + msg);
            return false;
        }
    }
    return true;
}

bool SqlTripStore::ensureTrip(const QString &tripId, QString *error)
{
    QSqlDatabase db = QSqlDatabase::database(m_connName);
    QSqlQuery check(db);
    check.prepare(QStringLiteral("SELECT 1 FROM trips WHERE trip_id=?"));
    check.addBindValue(tripId);
    if (check.exec() && check.next()) return true;

    QSqlQuery ins(db);
    ins.prepare(QStringLiteral("INSERT INTO trips(trip_id, vehicle_id, started_at, ended_at) "
                               "VALUES(?,?,?,?)"));
    ins.addBindValue(tripId);
    ins.addBindValue(QString());
    ins.addBindValue(QDateTime::currentDateTime().toString(Qt::ISODateWithMs));
    ins.addBindValue(QString());
    if (!ins.exec()) {
        const QString msg = QStringLiteral("写入行程失败: %1").arg(ins.lastError().text());
        if (error) *error = msg;
        LOG_WARN("[TripDataStore] " + msg);
        return false;
    }
    return true;
}

bool SqlTripStore::beginTrip(const QString &tripId, const QString &vehicleId,
                             const QDateTime &startTime, QString *error)
{
    if (!m_open) { if (error) *error = QStringLiteral("存储未打开"); return false; }

    QSqlDatabase db = QSqlDatabase::database(m_connName);
    QSqlQuery q(db);
    q.prepare(QStringLiteral("INSERT INTO trips(trip_id, vehicle_id, started_at, ended_at) "
                             "VALUES(?,?,?,?)"));
    q.addBindValue(tripId);
    q.addBindValue(vehicleId);
    q.addBindValue((startTime.isValid() ? startTime : QDateTime::currentDateTime())
                       .toString(Qt::ISODateWithMs));
    q.addBindValue(QString());
    if (!q.exec()) {
        // 已存在则退化为更新
        QSqlQuery u(db);
        u.prepare(QStringLiteral("UPDATE trips SET vehicle_id=? WHERE trip_id=?"));
        u.addBindValue(vehicleId);
        u.addBindValue(tripId);
        if (!u.exec()) {
            if (error) *error = u.lastError().text();
            LOG_WARN(QString("[TripDataStore] beginTrip 失败: %1").arg(u.lastError().text()));
            return false;
        }
    }

    m_liveOnboard[tripId] = 0;
    LOG_INFO(QString("[TripDataStore] 行程开始: trip=%1 vehicle=%2").arg(tripId, vehicleId));
    return true;
}

bool SqlTripStore::endTrip(const QString &tripId, const QDateTime &endTime, QString *error)
{
    if (!m_open) { if (error) *error = QStringLiteral("存储未打开"); return false; }

    QSqlDatabase db = QSqlDatabase::database(m_connName);
    QSqlQuery q(db);
    q.prepare(QStringLiteral("UPDATE trips SET ended_at=? WHERE trip_id=?"));
    q.addBindValue((endTime.isValid() ? endTime : QDateTime::currentDateTime())
                       .toString(Qt::ISODateWithMs));
    q.addBindValue(tripId);
    if (!q.exec()) {
        if (error) *error = q.lastError().text();
        LOG_WARN(QString("[TripDataStore] endTrip 失败: %1").arg(q.lastError().text()));
        return false;
    }
    LOG_INFO(QString("[TripDataStore] 行程结束: trip=%1").arg(tripId));
    return true;
}

int SqlTripStore::loadOnboardFromDb(const QString &tripId) const
{
    if (!m_open) return 0;
    QSqlDatabase db = QSqlDatabase::database(m_connName);
    QSqlQuery q(db);
    q.prepare(QStringLiteral("SELECT COALESCE(SUM(boarding),0) - COALESCE(SUM(alighting),0) "
                             "FROM stop_stats WHERE trip_id=?"));
    q.addBindValue(tripId);
    if (q.exec() && q.next()) return qMax(0, q.value(0).toInt());
    return 0;
}

bool SqlTripStore::applyStopEvent(const QString &tripId, const QString &stationId,
                                  const QString &stationName, const QString &passengerId,
                                  const QString &eventType, const QDateTime &time,
                                  QString *error)
{
    if (!m_open) { if (error) *error = QStringLiteral("存储未打开"); return false; }
    if (!ensureTrip(tripId, error)) return false;

    const QDateTime ts = time.isValid() ? time : QDateTime::currentDateTime();

    // ---- 内存累计 ----
    StopStat &st = m_liveStops[stopKey(tripId, stationId)];
    if (st.tripId.isEmpty()) { st.tripId = tripId; st.stationId = stationId; }
    if (!stationName.isEmpty()) st.stationName = stationName;

    int onboard = m_liveOnboard.contains(tripId) ? m_liveOnboard.value(tripId)
                                                 : loadOnboardFromDb(tripId);
    if (eventType == QLatin1String("BOARDING")) {
        st.boarding++;
        onboard++;
    } else {
        st.alighting++;
        if (onboard > 0) onboard--;
    }
    m_liveOnboard[tripId] = onboard;
    st.onboard = onboard;
    st.updatedAt = ts;

    // ---- 落库（事务）----
    QSqlDatabase db = QSqlDatabase::database(m_connName);
    db.transaction();

    QSqlQuery check(db);
    check.prepare(QStringLiteral("SELECT 1 FROM stop_stats WHERE trip_id=? AND station_id=?"));
    check.addBindValue(tripId);
    check.addBindValue(stationId);
    const bool exists = (check.exec() && check.next());

    QSqlQuery w(db);
    if (exists) {
        w.prepare(QStringLiteral("UPDATE stop_stats SET station_name=?, boarding=?, alighting=?, "
                                 "onboard=?, updated_at=? WHERE trip_id=? AND station_id=?"));
        w.addBindValue(st.stationName);
        w.addBindValue(st.boarding);
        w.addBindValue(st.alighting);
        w.addBindValue(onboard);
        w.addBindValue(ts.toString(Qt::ISODateWithMs));
        w.addBindValue(tripId);
        w.addBindValue(stationId);
    } else {
        w.prepare(QStringLiteral("INSERT INTO stop_stats(trip_id, station_id, station_name, "
                                 "boarding, alighting, onboard, updated_at) VALUES(?,?,?,?,?,?,?)"));
        w.addBindValue(tripId);
        w.addBindValue(stationId);
        w.addBindValue(st.stationName);
        w.addBindValue(st.boarding);
        w.addBindValue(st.alighting);
        w.addBindValue(onboard);
        w.addBindValue(ts.toString(Qt::ISODateWithMs));
    }
    if (!w.exec()) {
        db.rollback();
        const QString msg = QStringLiteral("写入站点统计失败: %1").arg(w.lastError().text());
        if (error) *error = msg;
        LOG_WARN("[TripDataStore] " + msg);
        return false;
    }

    QSqlQuery e(db);
    e.prepare(QStringLiteral("INSERT INTO passenger_events(trip_id, station_id, station_name, "
                             "passenger_id, event_type, recorded_at) VALUES(?,?,?,?,?,?)"));
    e.addBindValue(tripId);
    e.addBindValue(stationId);
    e.addBindValue(st.stationName);
    e.addBindValue(passengerId);
    e.addBindValue(eventType);
    e.addBindValue(ts.toString(Qt::ISODateWithMs));
    if (!e.exec()) {
        LOG_WARN(QString("[TripDataStore] 写入乘客事件失败(非致命): %1").arg(e.lastError().text()));
    }

    db.commit();

    LOG_INFO(QString("[TripDataStore] %1: trip=%2 站=%3(%4) 上车=%5 下车=%6 车厢内=%7")
                 .arg(eventType, tripId, stationId, st.stationName)
                 .arg(st.boarding).arg(st.alighting).arg(onboard));
    return true;
}

bool SqlTripStore::recordBoarding(const QString &tripId, const QString &stationId,
                                  const QString &stationName, const QString &passengerId,
                                  const QDateTime &time, QString *error)
{
    return applyStopEvent(tripId, stationId, stationName, passengerId,
                          QStringLiteral("BOARDING"), time, error);
}

bool SqlTripStore::recordAlighting(const QString &tripId, const QString &stationId,
                                   const QString &stationName, const QString &passengerId,
                                   const QDateTime &time, QString *error)
{
    return applyStopEvent(tripId, stationId, stationName, passengerId,
                          QStringLiteral("ALIGHTING"), time, error);
}

QVector<StopStat> SqlTripStore::stopStats(const QString &tripId) const
{
    QVector<StopStat> out;

    if (m_open) {
        QSqlDatabase db = QSqlDatabase::database(m_connName);
        QSqlQuery q(db);
        q.prepare(QStringLiteral("SELECT trip_id, station_id, station_name, boarding, alighting, "
                                 "onboard, updated_at FROM stop_stats WHERE trip_id=? "
                                 "ORDER BY updated_at ASC"));
        q.addBindValue(tripId);
        if (q.exec()) {
            while (q.next()) {
                StopStat s;
                s.tripId      = q.value(0).toString();
                s.stationId   = q.value(1).toString();
                s.stationName = q.value(2).toString();
                s.boarding    = q.value(3).toInt();
                s.alighting   = q.value(4).toInt();
                s.onboard     = q.value(5).toInt();
                s.updatedAt   = QDateTime::fromString(q.value(6).toString(), Qt::ISODateWithMs);
                out.append(s);
            }
            if (!out.isEmpty()) return out;
        }
    }

    // 回退：内存镜像
    for (auto it = m_liveStops.begin(); it != m_liveStops.end(); ++it) {
        if (it.value().tripId == tripId) out.append(it.value());
    }
    std::sort(out.begin(), out.end(), [](const StopStat &a, const StopStat &b) {
        return a.updatedAt < b.updatedAt;
    });
    return out;
}

int SqlTripStore::onboardCount(const QString &tripId) const
{
    if (m_liveOnboard.contains(tripId)) return m_liveOnboard.value(tripId);
    return loadOnboardFromDb(tripId);
}

StopStat SqlTripStore::currentStop(const QString &tripId) const
{
    StopStat best;
    bool found = false;
    for (auto it = m_liveStops.begin(); it != m_liveStops.end(); ++it) {
        const StopStat &s = it.value();
        if (s.tripId != tripId) continue;
        if (!found || s.updatedAt > best.updatedAt) { best = s; found = true; }
    }
    if (found) return best;

    const QVector<StopStat> all = stopStats(tripId);
    if (!all.isEmpty()) return all.last();
    return StopStat{};
}
