"""슈퍼챗/슈퍼스티커 금액을 config의 DONATION_TIERS 기준으로 분류."""
from config import config

_FALLBACK_TIER = {
    "tier": 1,
    "tnt_count": config.get("TNT_AMOUNT_ON_SUPERCHAT", 10),
    "mega": False,
    "mega_scale": 1,
    "shake_duration": 10,
    "shake_intensity": 10,
}


def _bucket_by_youtube_tier(yt_tier: int, tiers: list) -> dict:
    # 원화가 아닌 후원은 유튜브 자체 등급(1~수십)을 우리 티어 개수만큼 뭉뚱그려 매칭
    if not tiers:
        return _FALLBACK_TIER
    index = min(max(yt_tier - 1, 0) // 2, len(tiers) - 1)
    return tiers[index]


def classify(details: dict) -> dict:
    """superChatDetails 또는 superStickerDetails 딕셔너리를 받아 티어 설정을 반환."""
    tiers = config.get("DONATION_TIERS", [])
    if not tiers or not details:
        return _FALLBACK_TIER

    currency = details.get("currency", "")
    amount = int(details.get("amountMicros", 0)) / 1_000_000

    if currency == "KRW":
        for row in tiers:
            if amount >= row["min_krw"] and (row["max_krw"] is None or amount <= row["max_krw"]):
                return row
        return tiers[-1] if amount > tiers[-1]["min_krw"] else tiers[0]

    return _bucket_by_youtube_tier(int(details.get("tier", 1)), tiers)
