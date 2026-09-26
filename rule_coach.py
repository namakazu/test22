from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Finding:
    score: float
    title: str
    observed: str
    reason: str
    cue: str
    drill: str


def _num(metrics: Dict[str, Dict[str, float]], phase: str, key: str) -> Optional[float]:
    try:
        value = metrics.get(phase, {}).get(key)
        return float(value) if isinstance(value, (int, float)) else None
    except Exception:
        return None


def _avg(*values: Optional[float]) -> Optional[float]:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def _ratio_shift(a: Optional[float], b: Optional[float], scale: Optional[float]) -> Optional[float]:
    if a is None or b is None or scale is None or scale <= 0.001:
        return None
    return abs(b - a) / scale


def diagnose_rules(
    metrics: Dict[str, Dict[str, float]],
    view: str,
    handedness: str,
) -> str:
    """Return conservative beginner coaching from 2D pose metrics only."""
    findings: List[Finding] = []
    positives: List[str] = []

    address_lk = _num(metrics, "アドレス", "左膝角度(°)")
    address_rk = _num(metrics, "アドレス", "右膝角度(°)")
    address_knee = _avg(address_lk, address_rk)

    if address_knee is not None:
        if address_knee >= 172:
            findings.append(Finding(
                0.78,
                "構えで膝が伸び切り気味",
                f"アドレス時の膝角度は平均約{address_knee:.0f}°で、かなり真っすぐに見えます。",
                "膝がロックすると、体を回したときに上下動が出やすく、同じ位置にクラブを戻しにくくなります。",
                "『膝を曲げる』より『力を抜いて少しだけ沈む』",
                "クラブを持たずに構え、軽くジャンプして着地したときの膝の柔らかさを残したまま10回構える。",
            ))
        elif address_knee <= 125:
            findings.append(Finding(
                0.72,
                "構えで腰を落としすぎ気味",
                f"アドレス時の膝角度は平均約{address_knee:.0f}°で、深くしゃがんで見えます。",
                "腰を落としすぎると回転より上下動が増えやすく、スイングの再現性を作りにくくなります。",
                "『お尻を後ろ、膝は少しだけ』",
                "壁にお尻を軽く触れる位置で前傾を作り、膝は最後にほんの少し緩める動きを10回。",
            ))
        elif 138 <= address_knee <= 168:
            positives.append("アドレスの膝は、極端に伸び切ったりしゃがみ込んだりしていません。")

    lead_elbow_key = "左肘角度(°)" if handedness == "右打ち" else "右肘角度(°)"
    top_lead_elbow = _num(metrics, "トップ", lead_elbow_key)
    impact_lead_elbow = _num(metrics, "インパクト", lead_elbow_key)

    if top_lead_elbow is not None:
        if top_lead_elbow < 130:
            findings.append(Finding(
                0.63,
                "トップでリード腕が大きく曲がっている",
                f"トップでリード側の肘角度が約{top_lead_elbow:.0f}°です。2D映像上では曲がりが大きめです。",
                "初心者の場合、腕をたたみすぎると毎回同じ半径で振りにくくなることがあります。",
                "『腕を伸ばす』ではなく『胸から手を遠ざける』",
                "腰から腰のハーフスイングで、両手を胸からほぼ同じ距離に保つ意識で10球。",
            ))
        elif top_lead_elbow >= 148:
            positives.append("トップではリード腕が大きく折れず、スイング半径を保ちやすい形です。")

    if impact_lead_elbow is not None and impact_lead_elbow < 138:
        findings.append(Finding(
            0.58,
            "インパクト付近でリード腕が縮みやすい",
            f"インパクト付近のリード肘角度は約{impact_lead_elbow:.0f}°です。",
            "ボールに手で合わせにいく動きが強いと、当たり方が毎回変わりやすくなります。",
            "『ボールを打つ』より『体を回した先にボールがある』",
            "7〜9番アイアンで腰から腰の小さいスイングをし、ボールの先まで振り抜く練習を10球。",
        ))

    shoulder_width = _num(metrics, "アドレス", "肩幅(画面比)")
    addr_head_x = _num(metrics, "アドレス", "頭中心X")
    impact_head_x = _num(metrics, "インパクト", "頭中心X")
    addr_hip_x = _num(metrics, "アドレス", "腰中心X")
    top_hip_x = _num(metrics, "トップ", "腰中心X")

    if view.startswith("正面"):
        hip_sway = _ratio_shift(addr_hip_x, top_hip_x, shoulder_width)
        head_move = _ratio_shift(addr_head_x, impact_head_x, shoulder_width)

        if hip_sway is not None:
            if hip_sway > 0.72:
                findings.append(Finding(
                    min(0.95, 0.62 + hip_sway * 0.22),
                    "バックスイングで腰が横に流れ気味",
                    f"アドレスからトップまで、腰中心が肩幅の約{hip_sway:.1f}倍ぶん横へ移動しています。",
                    "横移動が大きいと、インパクトまでに元の位置へ戻す量が増え、当たりの再現性が落ちやすくなります。",
                    "『右へ動く』より『その場で胸を回す』",
                    "お尻を壁に軽く触れた状態で素振りし、左右へスライドせず回転する練習を10回。",
                ))
            elif hip_sway < 0.42:
                positives.append("トップまでの腰の横移動は大きくなく、回転を作りやすい範囲です。")

        if head_move is not None:
            if head_move > 0.62:
                findings.append(Finding(
                    min(0.90, 0.56 + head_move * 0.20),
                    "アドレスからインパクトまで頭が大きく横移動",
                    f"頭中心の横移動は肩幅の約{head_move:.1f}倍です。",
                    "頭を完全固定する必要はありませんが、大きく動くと最下点がずれやすくなります。",
                    "『頭を止める』ではなく『胸の中心を暴れさせない』",
                    "ティーをボールの位置に置き、腰から腰の素振りで胸の中心が左右へ流れすぎないか確認する。",
                ))
            elif head_move < 0.38:
                positives.append("アドレスからインパクトまで頭の横移動は比較的小さく見えます。")

    else:
        addr_lean = _num(metrics, "アドレス", "上体の傾き・鉛直基準(°)")
        impact_lean = _num(metrics, "インパクト", "上体の傾き・鉛直基準(°)")
        addr_head_y = _num(metrics, "アドレス", "頭中心Y")
        impact_head_y = _num(metrics, "インパクト", "頭中心Y")

        if addr_lean is not None:
            if addr_lean < 16:
                findings.append(Finding(
                    0.74,
                    "アドレスが起き気味",
                    f"後方映像での上体傾きは鉛直から約{addr_lean:.0f}°です。",
                    "前傾が浅すぎると、腕の通り道を作るためにスイング中に姿勢を変えやすくなります。",
                    "『腰を曲げる』ではなく『股関節からお辞儀』",
                    "お尻を後ろへ引きながら股関節から軽くお辞儀し、最後に膝を少し緩める動きを10回。",
                ))
            elif addr_lean > 52:
                findings.append(Finding(
                    0.72,
                    "アドレスで前に倒れすぎ気味",
                    f"後方映像での上体傾きは鉛直から約{addr_lean:.0f}°です。",
                    "前に倒れすぎるとバランスを保つための余計な動きが増え、再現性が落ちやすくなります。",
                    "『胸を少し起こして、足裏の真ん中で立つ』",
                    "つま先・かかとのどちらにも偏らない位置で構え、3秒静止できる姿勢を10回作る。",
                ))
            elif 22 <= addr_lean <= 45:
                positives.append("後方映像では、アドレスの前傾が極端に浅すぎたり深すぎたりしていません。")

        if addr_lean is not None and impact_lean is not None:
            lean_change = abs(impact_lean - addr_lean)
            if lean_change > 16:
                findings.append(Finding(
                    min(0.90, 0.60 + lean_change / 100),
                    "インパクトまでに前傾姿勢が大きく変わっている",
                    f"アドレスとインパクト付近で上体角度が約{lean_change:.0f}°変化しています。",
                    "姿勢が大きく変わると、クラブの最下点を毎回そろえるのが難しくなります。",
                    "『前傾を固定』ではなく『お尻の位置を保って回る』",
                    "お尻を壁に軽く触れた状態で、腰から腰の素振りを10回。壁から急に離れないようにする。",
                ))
            elif lean_change <= 10:
                positives.append("アドレスからインパクト付近まで、上体の前傾変化は比較的小さいです。")

        if addr_head_y is not None and impact_head_y is not None and shoulder_width:
            rise = (addr_head_y - impact_head_y) / shoulder_width
            if rise > 0.55:
                findings.append(Finding(
                    min(0.86, 0.55 + rise * 0.18),
                    "インパクトで体が起き上がり気味",
                    f"頭の高さが肩幅換算で約{rise:.1f}個ぶん上がっています。",
                    "早く起き上がるとクラブとの距離が変わり、トップや薄い当たりにつながりやすくなります。",
                    "『頭を下げる』ではなく『お尻を後ろに残す』",
                    "椅子の背や壁にお尻を軽く触れ、インパクト位置まで離れずに回る素振りを10回。",
                ))

    findings.sort(key=lambda x: x.score, reverse=True)
    top = findings[:2]

    lines: List[str] = []
    lines.append("## 今回の結論")
    if top:
        lines.append("次の10球では、下の項目だけ意識してください。数値は2D動画からの参考値です。")
    else:
        lines.append("大きく目立つ崩れは、この2D動画の範囲では検出されませんでした。まずは同じ構え・同じテンポで10球続けてください。")

    for i, f in enumerate(top, start=1):
        lines.append(f"\n## {i}. {f.title}")
        lines.append(f"- **今どう見えるか**: {f.observed}")
        lines.append(f"- **なぜ直すか**: {f.reason}")
        lines.append(f"- **次の10球の一言**: {f.cue}")
        lines.append(f"- **その場ドリル**: {f.drill}")

    lines.append("\n## 良かった点")
    if positives:
        for item in positives[:2]:
            lines.append(f"- {item}")
    else:
        lines.append("- 今回は改善点の抽出を優先しました。良し悪しを断定できない項目は評価していません。")

    lines.append("\n## この診断で見ていないもの")
    lines.append("- クラブフェース角、クラブ軌道、手首角度、3Dの肩・腰回転量は判定していません。")
    lines.append("- 正面/後方の1本の2D動画だけでは断定できない動きがあります。")
    lines.append("- 痛みが出る動作は中止してください。")

    return "\n".join(lines)
