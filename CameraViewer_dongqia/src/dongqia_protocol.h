#ifndef DONGQIA_PROTOCOL_H
#define DONGQIA_PROTOCOL_H

/**
 * @file dongqia_protocol.h
 * @brief 动恰3DV6 AI HTTP 新对接协议（客流时段聚合上行协议）数据结构定义
 *
 * 协议要点（摘自《动恰3DV6 AI HTTP 新对接协议20260911.pdf》）：
 *  - 设备 → 服务端，设备主动 POST，服务端只需实现接收端点
 *  - 每件报文 = 一分钟，按 (master_sn, batch_date, start_ts_s) 整件替换入库
 *  - 同一分钟会收到多次，用 revision 裁决，只保留 revision 更大的那一件
 *  - 服务端必须回 2xx 且响应体非空
 *  - 恒等式：raw_stats.count = final_stats.count + final_stats.repeat + non_customer.count
 *  - 各维度分桶之和 = 所属块的 count（not_measured 计入总数，不可并入 unknown）
 */

#include <QString>
#include <QStringList>
#include <QVector>
#include <QDateTime>
#include <QJsonArray>
#include <QJsonObject>
#include <opencv2/opencv.hpp>

namespace DongqiaFlow {

// ============================================================
// 分桶维度取值（行序固定，不可随意调整）
// ============================================================

namespace HeightCategory {
    static const QString AboveThreshold = QStringLiteral("above_threshold");
    static const QString BelowThreshold = QStringLiteral("below_threshold");
    static const QString Unknown        = QStringLiteral("unknown");
    static const QString NotMeasured    = QStringLiteral("not_measured");
}

namespace AgeRange {
    static const QString A0_9      = QStringLiteral("0_9");
    static const QString A10_16    = QStringLiteral("10_16");
    static const QString A17_30    = QStringLiteral("17_30");
    static const QString A31_45    = QStringLiteral("31_45");
    static const QString A46_60    = QStringLiteral("46_60");
    static const QString A61Plus   = QStringLiteral("61_plus");
    static const QString Unknown   = QStringLiteral("unknown");
    static const QString NotMeasured = QStringLiteral("not_measured");
}

namespace Gender {
    static const QString Male        = QStringLiteral("male");
    static const QString Female      = QStringLiteral("female");
    static const QString Unknown     = QStringLiteral("unknown");
    static const QString NotMeasured = QStringLiteral("not_measured");
}

namespace PersonType {
    static const QString Customer = QStringLiteral("customer"); // 仅内部使用，协议 person_type 桶里没有
    static const QString Staff    = QStringLiteral("staff");
    static const QString Rider    = QStringLiteral("rider");
    static const QString Courier  = QStringLiteral("courier");
}

namespace DwellRange {
    static const QString D0_1    = QStringLiteral("0_1");
    static const QString D1_5    = QStringLiteral("1_5");
    static const QString D5_15   = QStringLiteral("5_15");
    static const QString D15_30  = QStringLiteral("15_30");
    static const QString D30Plus = QStringLiteral("30_plus");
}

/// 人员类别（内部枚举，顾客走 final_stats，其余走 non_customer）
enum class PersonKind {
    Customer = 0,
    Staff,
    Rider,
    Courier
};

/// 把内部人员类别映射到协议 non_customer.person_type 取值
inline QString personKindToProtocol(PersonKind k)
{
    switch (k) {
    case PersonKind::Staff:   return PersonType::Staff;
    case PersonKind::Rider:   return PersonType::Rider;
    case PersonKind::Courier: return PersonType::Courier;
    default:                  return PersonType::Customer;
    }
}

// ============================================================
// 报文结构
// ============================================================

/// event_counts：门口四个方向的原始人次
struct EventCounts {
    int enter{0};       // 进入
    int leave{0};       // 离开
    int passBy{0};      // 路过（在门口区域出现但没进店）
    int turnBack{0};    // 折返（跨过门线后又退回去）
};

/// 单维度分桶行（height_category / person_type / dwell_range）
struct BucketRow {
    QString value;      // 维度取值
    int count{0};

    BucketRow() = default;
    BucketRow(const QString &v, int c) : value(v), count(c) {}
};

/// 年龄×性别交叉分桶行（age_range|gender）
struct AgeGenderRow {
    QString ageRange;
    QString gender;
    int count{0};

    AgeGenderRow() = default;
    AgeGenderRow(const QString &a, const QString &g, int c)
        : ageRange(a), gender(g), count(c) {}
};

/// 一件「时段聚合」报文 = 一分钟
struct IntervalAggregate {
    // ---- 信封 ----
    QString schemaVersion{QStringLiteral("1.0")};
    QString type{QStringLiteral("interval_aggregate")};
    QString masterSn;                    // 主相机 SN，一家店一个
    QStringList deviceSns;               // 当前设备清单
    QString tzName{QStringLiteral("Asia/Shanghai")};
    int utcOffsetSeconds{28800};         // 东为正
    bool isDst{false};
    QString batchDate;                   // 营业日 YYYY-MM-DD
    QString statBasis{QStringLiteral("enter")}; // "enter" 或 "leave"
    qint64 revision{0};                  // 这一版算出来的时刻（Unix 秒），只比大小
    int intervalS{60};
    qint64 startTsS{0};                  // 时段起点（含）
    qint64 endTsS{0};                    // 时段终点（不含）= start + 60

    // ---- 五个统计块 ----
    EventCounts eventCounts;

    int rawCount{0};                     // 基准方向进店次数（含员工骑手，不去重）
    QVector<BucketRow> rawHeight;        // raw_stats.height_category
    QVector<AgeGenderRow> rawAgeGender;  // raw_stats.age_range|gender

    int finalCount{0};                   // 本时段第一次进店的顾客数（人数）
    int finalRepeat{0};                  // 本时段回头人次（当天已来过）
    QVector<BucketRow> finalHeight;      // final_stats.height_category
    QVector<AgeGenderRow> finalAgeGender;// final_stats.age_range|gender

    int nonCustomerCount{0};             // 员工/骑手/快递出现人次
    QVector<BucketRow> nonCustomerPersonType; // non_customer.person_type

    int dwellCount{0};                   // 本时段已等到离开的顾客进店次数
    QVector<BucketRow> dwellRange;       // dwell_stats.dwell_range

    /// 幂等主键：master_sn + batch_date + start_ts_s
    QString idempotentKey() const
    {
        return QStringLiteral("interval|%1|%2|%3")
            .arg(masterSn, batchDate).arg(startTsS);
    }
};

/// 单人上下车/进出事件（用于「精准识别某一个人的上车/下车」）
struct PersonFlowEvent {
    QString masterSn;
    QString batchDate;
    QString statBasis{QStringLiteral("enter")};

    int visitorId{-1};                   // ReID 访客 ID（跨 track 稳定）
    PersonKind kind{PersonKind::Customer};
    QString eventType;                   // "boarding" / "alighting" / "enter" / "leave"
    QString direction;                   // "B->A"(上车) / "A->B"(下车) / ""(进出)

    qint64 enterTsS{0};                  // 进入通道时间
    qint64 exitTsS{0};                   // 离开通道时间
    qint64 eventTsS{0};                  // 事件发生时间（= exitTsS）
    int dwellSeconds{0};                 // 驻留时长（秒）

    float reidSimilarity{-1.0f};         // 身份置信度（与登记特征相似度）
    int regionId{-1};
    QString snapshotBase64;              // 事件抓拍 JPEG（Base64），可为空

    /// 幂等 key：单人单次事件
    QString idempotentKey() const
    {
        return QStringLiteral("person|%1|%2|%3|%4")
            .arg(masterSn).arg(visitorId).arg(eventType).arg(eventTsS);
    }
};

// ============================================================
// 工具函数
// ============================================================

/// 10 位 Unix 秒
inline qint64 toUnixSec(const QDateTime &dt)
{
    return dt.isValid() ? dt.toSecsSinceEpoch() : QDateTime::currentSecsSinceEpoch();
}

/// Unix 秒 → QDateTime（本地时区）
inline QDateTime fromUnixSec(qint64 sec)
{
    return QDateTime::fromSecsSinceEpoch(sec);
}

/// cv::Mat → JPEG Base64（用于事件抓拍上行）
QString matToJpegBase64(const cv::Mat &image, int quality = 85);

/// 计算驻留时长分桶（分钟，左闭右开；桶边界固定不可配置）
QString dwellBucketOfSeconds(int seconds);

} // namespace DongqiaFlow

#endif // DONGQIA_PROTOCOL_H
