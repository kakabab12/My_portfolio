"""'골드비' 이벤트 전용 장식용 파티클. 물리 충돌 없이 화면 위에서 금괴가 떨어지다 사라진다."""
import random
import pygame

from constants import INTERNAL_WIDTH

# 카메라는 세로(y)만 곡괭이를 따라가고 가로(x)는 거의 항상 0 근처라서, 월드 x 좌표가 곧 화면 x 좌표와 거의 같다.
# 그래서 곡괭이 위치를 중심으로 좁게 퍼뜨리는 대신, 화면 가로 전체 범위에서 바로 뿌린다.
_SPAWN_MARGIN = 30


class GoldCoin:
    def __init__(self, x, y, texture):
        self.x = x
        self.y = y
        self.texture = texture
        self.speed = random.uniform(300, 520)
        self.rotation = random.uniform(0, 360)
        self.spin_speed = random.uniform(-200, 200)
        self.age_ms = 0
        self.life_ms = 3200

    def update(self, dt_ms):
        self.y += self.speed * (dt_ms / 1000)
        self.rotation += self.spin_speed * (dt_ms / 1000)
        self.age_ms += dt_ms

    @property
    def finished(self):
        return self.age_ms >= self.life_ms

    def draw(self, screen, camera):
        rotated = pygame.transform.rotate(self.texture, self.rotation)
        rect = rotated.get_rect(center=(self.x - camera.offset_x, self.y - camera.offset_y))
        screen.blit(rotated, rect)


class GoldShower:
    def __init__(self, top_y, texture_atlas, atlas_items, count=26):
        rect = pygame.Rect(atlas_items["item"]["gold_ingot"])
        texture = texture_atlas.subsurface(rect)
        self.texture = pygame.transform.scale_by(texture, 2.5)

        self.coins = [
            GoldCoin(random.randint(_SPAWN_MARGIN, INTERNAL_WIDTH - _SPAWN_MARGIN), top_y - random.randint(0, 950), self.texture)
            for _ in range(count)
        ]

    def update(self, dt_ms):
        for coin in self.coins:
            coin.update(dt_ms)
        self.coins = [c for c in self.coins if not c.finished]

    def draw(self, screen, camera):
        for coin in self.coins:
            coin.draw(screen, camera)

    @property
    def finished(self):
        return len(self.coins) == 0


class ConfettiPiece:
    """텍스처 없이 색깔 있는 사각형만 그리는 가벼운 색종이 파티클."""

    COLORS = [
        (255, 99, 132), (54, 162, 235), (255, 206, 86),
        (75, 192, 192), (153, 102, 255), (255, 159, 64),
    ]

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.color = random.choice(self.COLORS)
        self.size = random.randint(10, 22)
        self.speed = random.uniform(260, 480)
        self.rotation = random.uniform(0, 360)
        self.spin_speed = random.uniform(-260, 260)
        self.age_ms = 0
        self.life_ms = 2600

        # 매 프레임 새로 만들지 않도록 사각형 base surface를 한 번만 만들어두고 회전만 매번 적용한다
        self.base_surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        pygame.draw.rect(self.base_surface, self.color, self.base_surface.get_rect())

    def update(self, dt_ms):
        self.y += self.speed * (dt_ms / 1000)
        self.rotation += self.spin_speed * (dt_ms / 1000)
        self.age_ms += dt_ms

    @property
    def finished(self):
        return self.age_ms >= self.life_ms

    def draw(self, screen, camera):
        rotated = pygame.transform.rotate(self.base_surface, self.rotation)
        rect = rotated.get_rect(center=(self.x - camera.offset_x, self.y - camera.offset_y))
        screen.blit(rotated, rect)


class ConfettiShower:
    """`confetti` 명령어용: 골드비와 같은 인터페이스를 쓰는 컬러풀 버전."""

    def __init__(self, top_y, count=45):
        self.pieces = [
            ConfettiPiece(random.randint(_SPAWN_MARGIN, INTERNAL_WIDTH - _SPAWN_MARGIN), top_y - random.randint(0, 950))
            for _ in range(count)
        ]

    def update(self, dt_ms):
        for piece in self.pieces:
            piece.update(dt_ms)
        self.pieces = [p for p in self.pieces if not p.finished]

    def draw(self, screen, camera):
        for piece in self.pieces:
            piece.draw(screen, camera)

    @property
    def finished(self):
        return len(self.pieces) == 0
