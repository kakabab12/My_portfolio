import math
import os
import pygame
from collections import deque

TIER_COLORS = {
    1: (140, 203, 255),
    2: (255, 217, 61),
    3: (255, 159, 67),
    4: (255, 71, 87),
}

# (표시 이름, 부제, 배경색, 아이콘 소스) - 작은 글씨에서 한글이 깨져 보여 전부 영어로 표기
# 아이콘 소스는 ("block"|"pickaxe", atlas_items 키) 또는 ("shape", "up"|"down"|"expand") 절차적 도형
COMMAND_LEGEND = [
    ("TNT", "Spawn TNT", (196, 64, 58), ("block", "tnt")),
    ("FAST", "Speed Up", (66, 176, 105), ("shape", "up")),
    ("SLOW", "Speed Down", (214, 130, 45), ("shape", "down")),
    ("BIG", "Enlarge Pickaxe", (68, 134, 214), ("shape", "expand")),
    ("WOOD", "Wood Pickaxe", (150, 111, 71), ("pickaxe", "wooden_pickaxe")),
    ("STONE", "Stone Pickaxe", (128, 128, 132), ("pickaxe", "stone_pickaxe")),
    ("IRON", "Iron Pickaxe", (200, 200, 205), ("pickaxe", "iron_pickaxe")),
    ("GOLD", "Gold Pickaxe", (232, 191, 62), ("pickaxe", "golden_pickaxe")),
    ("DIAMOND", "Diamond Pickaxe", (86, 210, 216), ("pickaxe", "diamond_pickaxe")),
    ("NETHERITE", "Netherite Pickaxe", (94, 78, 100), ("pickaxe", "netherite_pickaxe")),
    ("DONATE", "Tiered Bomb Effect", (211, 84, 158), ("block", "mega_tnt")),
    ("NUKE", "Giant Explosion", (231, 76, 60), ("block", "mega_tnt")),
    ("MISSILE", "Missile Barrage", (230, 126, 34), ("block", "tnt")),
    ("METEOR", "Flaming Meteor", (192, 57, 43), ("block", "mega_tnt")),
    ("EARTHQUAKE", "Screen Shake", (142, 68, 173), ("shape", "quake")),
    ("GOLD RAIN", "Coin Shower", (241, 196, 15), ("item", "gold_ingot")),
    ("FREEZE", "Stop Falling", (100, 181, 246), ("shape", "snowflake")),
    ("TINY", "Shrink Pickaxe", (154, 205, 255), ("shape", "shrink")),
    ("CONFETTI", "Color Party", (230, 126, 194), ("shape", "sparkle")),
    ("LUCKY", "Bonus Ores", (46, 204, 113), ("item", "diamond")),
    ("SLOWMO", "Extreme Slow-Mo", (93, 109, 219), ("shape", "down")),
]

_KOREAN_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "malgungothic",
    "applegothic",
    "notosanscjkkr",
    "notosanskr",
]


def load_korean_font(size):
    """Windows 기본 폰트(pygame.font.Font(None, ...))는 한글 글리프가 없어 텍스트가 깨진다.
    맑은 고딕 등 한글이 있는 폰트를 우선 찾아서 로드하고, 못 찾으면 기본 폰트로 대체한다."""
    for candidate in _KOREAN_FONT_CANDIDATES:
        try:
            if candidate.endswith(".ttf"):
                if os.path.exists(candidate):
                    return pygame.font.Font(candidate, size)
                continue
            font = pygame.font.SysFont(candidate, size)
            if font is not None:
                return font
        except Exception:
            continue
    return pygame.font.Font(None, size)


def _smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def render_text_with_outline(text, font, text_color, outline_color, outline_width=2):
    # Render the text in the main color.
    text_surface = font.render(text, True, text_color)
    # Create a new surface larger than the text surface to hold the outline.
    w, h = text_surface.get_size()
    outline_surface = pygame.Surface((w + 2*outline_width, h + 2*outline_width), pygame.SRCALPHA)

    # Blit the text multiple times in the outline color, offset by outline_width in every direction.
    for dx in range(-outline_width, outline_width+1):
        for dy in range(-outline_width, outline_width+1):
            # Only draw outline if offset is non-zero (avoids overdraw, though it's not a big deal)
            if dx != 0 or dy != 0:
                pos = (dx + outline_width, dy + outline_width)
                outline_surface.blit(font.render(text, True, outline_color), pos)

    # Blit the main text in the center.
    outline_surface.blit(text_surface, (outline_width, outline_width))
    return outline_surface

class Hud:
    def __init__(self, texture_atlas, atlas_items, position=(32, 32)):
        """
        :param texture_atlas: The atlas surface containing the item icons.
        :param atlas_items: A dict with keys under "item" for each ore.
        :param position: Top-left position where the HUD will be drawn.
        """
        self.texture_atlas = texture_atlas
        self.atlas_items = atlas_items

        # Initialize ore amounts to 0.
        self.amounts = {
            "coal": 0,
            "iron_ingot": 0,
            "copper_ingot": 0,
            "gold_ingot": 0,
            "redstone": 0,
            "lapis_lazuli": 0,
            "diamond": 0,
            "emerald": 0,
        }

        self.position = position
        self.icon_size = (64, 64)  # Size to draw each icon
        self.spacing = 15  # Space between items

        # Initialize a font (using the default font and size 24)
        self.font = pygame.font.Font(None, 64)
        self.icon_cache = {}
        for ore in self.amounts:
            if ore in self.atlas_items["item"]:
                icon_rect = pygame.Rect(self.atlas_items["item"][ore])
                icon = self.texture_atlas.subsurface(icon_rect)
                icon = pygame.transform.scale(icon, self.icon_size)
                self.icon_cache[ore] = icon

        self.amount_text_cache = {}

        # 후원 알림 배너 + 최근 댓글 명령 피드 (한글이 섞이므로 한글 지원 폰트 사용)
        self.alert_font = load_korean_font(84)
        self.feed_font = load_korean_font(34)
        self.banners = []
        self.flashes = []
        self._banner_base_cache_key = None
        self._banner_base_surface = None
        self.command_feed = deque(maxlen=6)
        self._feed_cache_key = None
        self._feed_surfaces = []

        # 명령어 안내판 (항상 같은 내용이라 한 번만 그려서 캐싱)
        self.legend_title_font = load_korean_font(26)
        self.legend_label_font = load_korean_font(18)
        self.legend_sub_font = load_korean_font(13)
        self.command_panel_surface = self._build_command_panel()

        # 다음 랜덤 이벤트까지 남은 시간 표시
        self.timer_font = load_korean_font(24)
        self.event_timers = []
        self._timer_cache_key = None
        self._timer_panel_surface = None

        # 콤보 카운터 (연속으로 블록을 부수면 쌓임, 화면 상단 중앙에 크게 표시)
        self.combo_font = load_korean_font(56)
        self.best_combo_font = load_korean_font(22)
        self.combo_count = 0
        self.best_combo = 0
        self._combo_cache_key = None
        self._combo_surfaces = None
        self._combo_pulse_end = 0
        self._combo_pulse_duration_ms = 180

    def _build_icon_surface(self, icon_source, size):
        kind, key = icon_source
        box = pygame.Surface((size, size), pygame.SRCALPHA)

        if kind in ("block", "pickaxe", "item") and key in self.atlas_items.get(kind, {}):
            rect = pygame.Rect(self.atlas_items[kind][key])
            icon = self.texture_atlas.subsurface(rect)
            target = int(size * 0.8)
            icon = pygame.transform.smoothscale(icon, (target, target))
            box.blit(icon, ((size - target) // 2, (size - target) // 2))
            return box

        color = (255, 255, 255)
        cx, cy, s = size / 2, size / 2, size * 0.28
        if key == "up":
            pygame.draw.polygon(box, color, [(cx, cy - s), (cx - s, cy + s * 0.7), (cx + s, cy + s * 0.7)])
        elif key == "down":
            pygame.draw.polygon(box, color, [(cx, cy + s), (cx - s, cy - s * 0.7), (cx + s, cy - s * 0.7)])
        elif key == "expand":
            pygame.draw.polygon(box, color, [(cx - s, cy), (cx, cy - s), (cx + s, cy)])
            pygame.draw.polygon(box, color, [(cx - s, cy), (cx, cy + s), (cx + s, cy)])
        elif key == "quake":
            zigzag = [(cx - s, cy - s), (cx - s * 0.2, cy - s * 0.1), (cx + s * 0.3, cy - s * 0.6),
                      (cx, cy + s * 0.2), (cx + s * 0.5, cy + s), (cx - s * 0.1, cy + s * 0.3)]
            pygame.draw.lines(box, color, False, zigzag, width=max(2, size // 12))
        elif key == "shrink":
            # BIG의 확장 화살표(expand)와 반대로, 안쪽을 향하는 작은 다이아몬드
            pygame.draw.polygon(box, color, [(cx, cy - s * 0.5), (cx - s * 0.5, cy), (cx, cy + s * 0.5), (cx + s * 0.5, cy)])
        elif key == "snowflake":
            for angle in (0, 60, 120):
                rad = math.radians(angle)
                dx, dy = math.cos(rad) * s, math.sin(rad) * s
                pygame.draw.line(box, color, (cx - dx, cy - dy), (cx + dx, cy + dy), width=max(2, size // 14))
        elif key == "sparkle":
            points = []
            for i in range(8):
                radius = s if i % 2 == 0 else s * 0.35
                angle = math.radians(i * 45)
                points.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
            pygame.draw.polygon(box, color, points)
        return box

    def _build_command_panel(self):
        icon_size = 34
        row_height = 46
        row_gap = 6
        padding = 12
        icon_gap = 10

        title_surface = render_text_with_outline("COMMAND", self.legend_title_font, (255, 255, 255), (0, 0, 0), outline_width=2)

        rows = []
        max_text_width = title_surface.get_width()
        for label, subtitle, color, icon_source in COMMAND_LEGEND:
            label_surface = render_text_with_outline(label, self.legend_label_font, (255, 255, 255), (0, 0, 0), outline_width=1)
            subtitle_surface = render_text_with_outline(subtitle, self.legend_sub_font, (235, 235, 235), (0, 0, 0), outline_width=1)
            rows.append((label_surface, subtitle_surface, color, icon_source))
            max_text_width = max(max_text_width, label_surface.get_width(), subtitle_surface.get_width())

        panel_width = padding * 2 + max(title_surface.get_width(), icon_size + icon_gap + max_text_width)
        panel_height = padding * 2 + title_surface.get_height() + row_gap + len(rows) * (row_height + row_gap)

        # 입체감을 위한 그림자 여백을 두고 캔버스를 살짝 더 크게 잡는다
        shadow_offset = (5, 7)
        canvas = pygame.Surface((panel_width + shadow_offset[0], panel_height + shadow_offset[1]), pygame.SRCALPHA)

        shadow = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        pygame.draw.rect(shadow, (0, 0, 0, 110), shadow.get_rect(), border_radius=12)
        canvas.blit(shadow, shadow_offset)

        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)

        pygame.draw.rect(panel, (8, 9, 16, 210), panel.get_rect(), border_radius=12)
        pygame.draw.rect(panel, (255, 255, 255, 35), panel.get_rect(), width=1, border_radius=12)

        title_rect = title_surface.get_rect(midtop=(panel_width // 2, padding))
        panel.blit(title_surface, title_rect)

        y = padding + title_surface.get_height() + row_gap

        for label_surface, subtitle_surface, color, icon_source in rows:
            row_rect = pygame.Rect(padding, y, panel_width - padding * 2, row_height)
            pygame.draw.rect(panel, (*color, 235), row_rect, border_radius=9)

            # 살짝 광택 나는 느낌을 주는 상단부 하이라이트
            gloss_rect = pygame.Rect(row_rect.x + 2, row_rect.y + 2, row_rect.width - 4, max(1, row_rect.height // 2 - 2))
            gloss = pygame.Surface(gloss_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(gloss, (255, 255, 255, 55), gloss.get_rect(), border_radius=7)
            panel.blit(gloss, gloss_rect.topleft)

            pygame.draw.rect(panel, (255, 255, 255, 60), row_rect, width=1, border_radius=9)

            icon = self._build_icon_surface(icon_source, icon_size)
            icon_rect = icon.get_rect(midleft=(row_rect.x + 8, row_rect.centery))
            panel.blit(icon, icon_rect)

            text_x = icon_rect.right + icon_gap
            label_rect = label_surface.get_rect(topleft=(text_x, row_rect.y + 4))
            subtitle_rect = subtitle_surface.get_rect(topleft=(text_x, label_rect.bottom))
            panel.blit(label_surface, label_rect)
            panel.blit(subtitle_surface, subtitle_rect)

            y += row_height + row_gap

        canvas.blit(panel, (0, 0))
        return canvas

    def add_event_banner(self, text, color, duration_ms=3000):
        """화면 상단 중앙에 큰 배너를 띄운다. 후원 알림뿐 아니라 재미 커맨드 이벤트에도 재사용."""
        now = pygame.time.get_ticks()
        self.banners.append({
            "text": text,
            "color": color,
            "start": now,
            "expire": now + duration_ms,
        })

    def add_donation_alert(self, author, amount_text, tier):
        """슈퍼챗/슈퍼스티커가 들어오면 화면 상단에 티어별 색상으로 배너를 띄운다."""
        color = TIER_COLORS.get(tier, (255, 255, 255))
        self.add_event_banner(f"{author} 님이 {amount_text} 후원!", color, 2500 + tier * 1000)

    def add_screen_flash(self, color=(255, 255, 255), duration_ms=220, peak_alpha=210):
        """핵/메가 도네이션처럼 임팩트가 큰 순간에 화면 전체를 짧게 번쩍이게 한다."""
        self.flashes.append({
            "color": color,
            "start": pygame.time.get_ticks(),
            "duration": duration_ms,
            "peak_alpha": peak_alpha,
        })

    def add_command_feed(self, author, command_text):
        """채팅 명령이 실제로 적용될 때마다 화면 하단에 누가 무엇을 했는지 표시한다."""
        self.command_feed.append(f"{author}: {command_text}")

    def set_combo(self, count, best):
        """count: 현재 연속 콤보 수, best: 이번 방송 세션 중 최고 기록"""
        if count > self.combo_count:
            self._combo_pulse_end = pygame.time.get_ticks() + self._combo_pulse_duration_ms
        self.combo_count = count
        self.best_combo = best

    def _draw_combo(self, screen):
        if self.combo_count < 2:
            return

        cache_key = (self.combo_count, self.best_combo)
        if cache_key != self._combo_cache_key:
            self._combo_cache_key = cache_key

            if self.combo_count >= 100:
                color = (255, 71, 87)
            elif self.combo_count >= 50:
                color = (255, 159, 67)
            elif self.combo_count >= 25:
                color = (255, 217, 61)
            elif self.combo_count >= 10:
                color = (140, 203, 255)
            else:
                color = (255, 255, 255)

            combo_surface = render_text_with_outline(f"COMBO x{self.combo_count}", self.combo_font, color, (0, 0, 0), outline_width=3)
            best_surface = None
            if self.best_combo > 0:
                best_surface = render_text_with_outline(f"BEST {self.best_combo}", self.best_combo_font, (225, 225, 225), (0, 0, 0), outline_width=1)
            self._combo_surfaces = (combo_surface, best_surface)

        combo_surface, best_surface = self._combo_surfaces

        # 콤보가 오를 때마다 살짝 팍 커졌다가 원래 크기로 줄어드는 펀치감
        pulse_remaining = self._combo_pulse_end - pygame.time.get_ticks()
        if pulse_remaining > 0:
            t = _smoothstep(pulse_remaining / self._combo_pulse_duration_ms)  # 1 -> 0
            scale = 1.0 + t * 0.35
            w, h = combo_surface.get_size()
            combo_surface = pygame.transform.smoothscale(combo_surface, (max(1, int(w * scale)), max(1, int(h * scale))))

        combo_rect = combo_surface.get_rect(midtop=(screen.get_width() // 2, 24))
        screen.blit(combo_surface, combo_rect)
        if best_surface:
            best_rect = best_surface.get_rect(midtop=(screen.get_width() // 2, combo_rect.bottom + 2 + (combo_rect.height - self._combo_surfaces[0].get_height())))
            screen.blit(best_surface, best_rect)

    def set_event_timers(self, timers):
        """timers: (라벨, 남은 초) 튜플 리스트. 남은 초가 None이면 그 항목은 표시하지 않는다."""
        self.event_timers = timers

    def _draw_event_timers(self, screen, position):
        display_rows = [(label, max(seconds, 0)) for label, seconds in self.event_timers if seconds is not None]
        if not display_rows:
            return

        cache_key = tuple((label, round(seconds, 1)) for label, seconds in display_rows)
        if cache_key != self._timer_cache_key:
            self._timer_cache_key = cache_key
            padding = 12
            row_gap = 4

            row_surfaces = [
                render_text_with_outline(f"{label}  {seconds:4.1f}s", self.timer_font, (255, 255, 255), (0, 0, 0), outline_width=1)
                for label, seconds in display_rows
            ]
            panel_width = max(r.get_width() for r in row_surfaces) + padding * 2
            panel_height = sum(r.get_height() for r in row_surfaces) + row_gap * (len(row_surfaces) - 1) + padding * 2

            panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
            pygame.draw.rect(panel, (8, 9, 16, 200), panel.get_rect(), border_radius=10)
            pygame.draw.rect(panel, (255, 255, 255, 40), panel.get_rect(), width=1, border_radius=10)

            ry = padding
            for row_surface in row_surfaces:
                panel.blit(row_surface, (padding, ry))
                ry += row_surface.get_height() + row_gap

            self._timer_panel_surface = panel

        screen.blit(self._timer_panel_surface, position)

    def update_amounts(self, new_amounts):
        """
        Update the ore amounts.
        :param new_amounts: Dict with ore names as keys and integer amounts as values.
        """
        self.amounts.update(new_amounts)

    def draw(self, screen):
        """
        Draws the HUD: each ore icon with its amount and other indicators.
        """
        x, y = self.position

        for ore, amount in self.amounts.items():
            # Retrieve the icon rect from atlas_items["item"][ore]
            if ore in self.icon_cache:
                screen.blit(self.icon_cache[ore], (x, y))
            else:
                # In case the ore key is missing, skip drawing the icon
                continue

            text_surface = self.amount_text_cache.get(ore)
            if text_surface is None or text_surface[0] != amount:
                text = str(amount)
                text_surface = (amount, render_text_with_outline(text, self.font, (255, 255, 255), (0, 0, 0), outline_width=2))
                self.amount_text_cache[ore] = text_surface

            # Position text to the right of the icon
            text_x = x + self.icon_size[0] + self.spacing
            text_y = y + (self.icon_size[1] - text_surface[1].get_height()) // 2 + 3
            screen.blit(text_surface[1], (text_x, text_y))

            # Move to the next line
            y += self.icon_size[1] + self.spacing

        # 다음 랜덤 이벤트까지 남은 시간 (기존 Y 좌표/Normal 표시 자리에 대신 표시)
        self._draw_event_timers(screen, (x + self.spacing, y + self.spacing))

        # 콤보 카운터 (화면 상단 중앙)
        self._draw_combo(screen)

        # 명령어 안내판 (화면 맨 오른쪽, 세로로 도킹, 항상 같은 내용)
        panel_x = screen.get_width() - self.command_panel_surface.get_width() - 16
        panel_y = 24
        screen.blit(self.command_panel_surface, (panel_x, panel_y))

        # 이벤트 배너 (후원 알림 + 재미 커맨드 알림, 확대/페이드 애니메이션과 함께 노출)
        now = pygame.time.get_ticks()
        self.banners = [b for b in self.banners if b["expire"] > now]
        if self.banners:
            banner = self.banners[0]
            elapsed = now - banner["start"]
            remaining = banner["expire"] - now
            fade_in_ms = 220
            fade_out_ms = 350

            if elapsed < fade_in_ms:
                t = _smoothstep(elapsed / fade_in_ms)
                scale = 0.7 + 0.3 * t
                alpha = int(255 * t)
            elif remaining < fade_out_ms:
                t = _smoothstep(remaining / fade_out_ms)
                scale = 1.0 + (1 - t) * 0.1
                alpha = int(255 * t)
            else:
                scale = 1.0
                alpha = 255

            # 텍스트 자체는 배너가 바뀔 때만 다시 렌더링 (아웃라인 렌더는 프레임당 9번 font.render를 호출해서 꽤 무겁다)
            banner_cache_key = (banner["text"], banner["color"])
            if banner_cache_key != self._banner_base_cache_key:
                self._banner_base_cache_key = banner_cache_key
                self._banner_base_surface = render_text_with_outline(banner["text"], self.alert_font, banner["color"], (0, 0, 0), outline_width=3)

            banner_surface = self._banner_base_surface
            if abs(scale - 1.0) > 0.001:
                w, h = banner_surface.get_size()
                banner_surface = pygame.transform.smoothscale(banner_surface, (max(1, int(w * scale)), max(1, int(h * scale))))
            banner_surface.set_alpha(max(0, min(255, alpha)))
            banner_rect = banner_surface.get_rect(center=(screen.get_width() // 2, 140))
            screen.blit(banner_surface, banner_rect)

        # 최근 댓글 명령 피드 (화면 하단 왼쪽). 내용이 바뀔 때만 다시 렌더링해서 매 프레임 재렌더를 피한다.
        feed_key = tuple(self.command_feed)
        if feed_key != self._feed_cache_key:
            self._feed_cache_key = feed_key
            self._feed_surfaces = [
                render_text_with_outline(entry, self.feed_font, (255, 255, 255), (0, 0, 0), outline_width=1)
                for entry in self.command_feed
            ]

        feed_x = 20
        feed_y = screen.get_height() - 40 - (len(self._feed_surfaces) * 34)
        for entry_surface in self._feed_surfaces:
            screen.blit(entry_surface, (feed_x, feed_y))
            feed_y += 34

        # 화면 플래시 (핵/메가 도네이션 등 임팩트가 큰 이벤트에서 화면 전체를 덮어 번쩍임)
        self.flashes = [f for f in self.flashes if now - f["start"] < f["duration"]]
        for flash in self.flashes:
            t = (now - flash["start"]) / flash["duration"]
            alpha = int(flash["peak_alpha"] * (1 - _smoothstep(t)))
            if alpha <= 0:
                continue
            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            overlay.fill((*flash["color"], alpha))
            screen.blit(overlay, (0, 0))
