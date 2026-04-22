"""PySide6-интерфейс для запуска и просмотра анализа версий ИИС."""

from __future__ import annotations

import json
import logging
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QThread, Qt, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QTableView,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from config.settings import (
    DATASET_DEFAULTS,
    DYNAMIC_WINDOW_DEFAULTS,
    OUTPUT_DIR,
    PLOTS_DIR,
    PROCESSING_MODES,
    SUPPORTED_DATASETS,
    ensure_runtime_directories,
)
from run_analysis import configure_logging, run_dataset
from models.intervention_analysis import IISInterventionSimulator

LOGGER = logging.getLogger("iis_gui")
MODEL_ORDER = ("IISVersion1", "IISVersion2", "IISVersion3", "IISVersion4", "IISVersion5", "IISVersion6", "IISVersion7")
LIVE_LABEL_COLORS = {
    "baseline": QColor("#2E7D32"),
    "disbalance": QColor("#F9A825"),
    "stress": QColor("#C62828"),
    "amusement": QColor("#1565C0"),
    "unknown": QColor("#546E7A"),
}
SEGMENT_FEATURE_COLUMNS = (
    "valence",
    "arousal",
    "dominance",
    "liking",
    "stress_label",
    "eeg_left_power",
    "eeg_right_power",
    "eeg_gamma_power",
    "eeg_alpha_left",
    "eeg_alpha_right",
    "alpha_asymmetry",
    "gamma_alpha_ratio",
    "hrv_hf",
    "hrv_lf",
    "hrv_hf_lf",
    "hrv_lf_hf",
    "heart_rate",
    "hrv_rmssd",
    "hrv_sdnn",
    "direct_feature_count",
    "proxy_feature_count",
    "unavailable_feature_count",
)
VERSION_RESULT_COLUMNS = ("version", "IIS", "RES", "state_map_4", "A", "Gamma", "H", "V", "Q", "K", "coverage_ratio", "active_component_count")
LIVE_PREVIEW_COLUMNS = ("window_start_sec", "window_end_sec", "label", "IIS", "RES", "state_map_4", "A", "Gamma", "V", "Q", "H", "K")
DYNAMIC_PREVIEW_COLUMNS = (
    "time_sec",
    "label",
    "IIS",
    "RES",
    "IIS_fast",
    "IIS_smooth_core",
    "IIS_dynamic",
    "RES_dynamic",
    "state_map_4_dynamic",
    "stress_drive",
    "response_gain",
    "recovery_gain",
    "IIS_volatility",
    "dynamic_mode",
)
STATIC_PLOT_SPECS = (
    ("boxplot_iis", "Boxplot IIS по классам"),
    ("stress_baseline_diff", "Разница stress-baseline"),
    ("scatter_q_valence", "Q vs valence"),
    ("scatter_iis_arousal", "IIS vs arousal"),
    ("component_contributions", "Средние вклады компонентов"),
    ("sensitivity", "Чувствительность компонентов"),
)
INTERVENTION_TARGETS = (
    "heart_rate",
    "hrv_hf_lf",
    "hrv_lf_hf",
    "alpha_asymmetry",
    "gamma_alpha_ratio",
    "eeg_gamma_power",
    "valence",
    "arousal",
)


def live_label_color(label_value: Any) -> QColor:
    """Возвращает цвет точки по имени класса/состояния."""

    label_name = str(label_value or "unknown").lower()
    return LIVE_LABEL_COLORS.get(label_name, LIVE_LABEL_COLORS["unknown"])


def safe_json_loads(value: Any) -> Any:
    """Безопасно разбирает JSON-строку."""

    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def format_float(value: Any, digits: int = 5) -> str:
    """Преобразует число к компактной строке."""

    try:
        numeric = float(value)
    except Exception:
        return ""
    if pd.isna(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def build_segment_key(row: pd.Series | dict[str, Any]) -> str:
    """Строит читаемый ключ сегмента для таблиц и селектора."""

    if isinstance(row, pd.Series):
        payload = row.to_dict()
    else:
        payload = dict(row)

    subject = str(payload.get("subject_id", "") or "unknown")
    segment = str(payload.get("segment_id", "") or "segment")
    label = str(payload.get("label", "") or "no-label")
    source_record = str(payload.get("source_record_id", "") or "")

    start_value = pd.to_numeric(payload.get("window_start_sec"), errors="coerce")
    end_value = pd.to_numeric(payload.get("window_end_sec"), errors="coerce")
    if pd.notna(start_value) and pd.notna(end_value):
        window_label = f"{float(start_value):.1f}-{float(end_value):.1f} c"
    else:
        window_label = "без окна"

    parts = [subject, segment, label, window_label]
    if source_record:
        parts.append(source_record)
    return " | ".join(parts)


class DataFrameTableModel(QAbstractTableModel):
    """Простой адаптер pandas.DataFrame для QTableView."""

    def __init__(self, frame: pd.DataFrame | None = None) -> None:
        super().__init__()
        self._frame = frame.copy() if frame is not None else pd.DataFrame()

    def set_frame(self, frame: pd.DataFrame | None) -> None:
        """Заменяет содержимое таблицы."""

        self.beginResetModel()
        self._frame = frame.copy() if frame is not None else pd.DataFrame()
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        """Возвращает число строк."""

        if parent.isValid():
            return 0
        return len(self._frame.index)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        """Возвращает число столбцов."""

        if parent.isValid():
            return 0
        return len(self._frame.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        """Возвращает данные ячейки."""

        if not index.isValid():
            return None

        value = self._frame.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            if pd.isna(value):
                return ""
            if isinstance(value, float):
                return format_float(value)
            return str(value)

        if role == Qt.TextAlignmentRole:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(Qt.AlignRight | Qt.AlignVCenter)
            return int(Qt.AlignLeft | Qt.AlignVCenter)

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        """Возвращает заголовки строк и столбцов."""

        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            if section < len(self._frame.columns):
                return str(self._frame.columns[section])
            return ""
        if section < len(self._frame.index):
            return str(section + 1)
        return ""


class PlotViewer(QWidget):
    """Виджет для показа сохранённого PNG-файла."""

    def __init__(self, title: str) -> None:
        super().__init__()
        self._image_path: Path | None = None
        self._title_label = QLabel(title)
        self._title_label.setWordWrap(True)
        self._path_label = QLabel("График не загружен.")
        self._path_label.setWordWrap(True)
        self._image_label = QLabel("Нет изображения.")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumHeight(260)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self._image_label)

        open_button = QPushButton("Открыть PNG")
        open_button.clicked.connect(self.open_image)

        layout = QVBoxLayout(self)
        layout.addWidget(self._title_label)
        layout.addWidget(self._path_label)
        layout.addWidget(scroll_area, stretch=1)
        layout.addWidget(open_button)

    def set_image(self, image_path: Path | None) -> None:
        """Обновляет показываемое изображение."""

        self._image_path = image_path
        if image_path is None or not image_path.exists():
            self._path_label.setText("График не найден.")
            self._image_label.setPixmap(QPixmap())
            self._image_label.setText("Нет изображения.")
            return

        self._path_label.setText(str(image_path))
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self._image_label.setPixmap(QPixmap())
            self._image_label.setText("Не удалось прочитать PNG.")
            return
        scaled = pixmap.scaledToWidth(960, Qt.SmoothTransformation) if pixmap.width() > 960 else pixmap
        self._image_label.setPixmap(scaled)
        self._image_label.setText("")

    def open_image(self) -> None:
        """Открывает текущий PNG внешним приложением."""

        if self._image_path is None or not self._image_path.exists():
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._image_path)))


class LiveScatterWidget(QWidget):
    """Простое живое 2D-облако точек для предпросмотра во время расчёта."""

    LABEL_COLORS = {
        "baseline": QColor("#2E7D32"),
        "disbalance": QColor("#F9A825"),
        "stress": QColor("#C62828"),
        "amusement": QColor("#1565C0"),
        "unknown": QColor("#546E7A"),
    }

    def __init__(self) -> None:
        super().__init__()
        self._frame = pd.DataFrame()
        self._x_column = "IIS"
        self._y_column = "RES"
        self._title = "Живое облако состояний"
        self._selected_row_index: int | None = None
        self.setMinimumHeight(320)

    def set_preview(self, frame: pd.DataFrame, *, x_column: str, y_column: str, title: str) -> None:
        """Обновляет живой предпросмотр точек."""

        self._frame = frame.copy() if frame is not None else pd.DataFrame()
        self._x_column = x_column
        self._y_column = y_column
        self._title = title
        self.update()

    def set_selected_row_index(self, row_index: int | None) -> None:
        """Подсвечивает выбранную точку по индексу preview-строки."""

        self._selected_row_index = row_index if row_index in self._frame.index else None
        self.update()

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        """Рисует точки в плоскости двух выбранных координат."""

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))

        margin_left = 56
        margin_right = 18
        margin_top = 38
        margin_bottom = 36
        plot_rect = self.rect().adjusted(margin_left, margin_top, -margin_right, -margin_bottom)

        painter.setPen(QPen(QColor("#263238"), 1))
        painter.drawText(12, 24, self._title)

        if self._frame.empty or self._x_column not in self._frame.columns or self._y_column not in self._frame.columns:
            painter.setPen(QColor("#546E7A"))
            painter.drawRect(plot_rect)
            painter.drawText(plot_rect, int(Qt.AlignCenter), "Жду первые точки…")
            return

        x_values = pd.to_numeric(self._frame[self._x_column], errors="coerce")
        y_values = pd.to_numeric(self._frame[self._y_column], errors="coerce")
        valid = x_values.notna() & y_values.notna()
        if not bool(valid.any()):
            painter.setPen(QColor("#546E7A"))
            painter.drawRect(plot_rect)
            painter.drawText(plot_rect, int(Qt.AlignCenter), "Пока нет валидных координат.")
            return

        frame = self._frame.loc[valid].copy()
        x_values = x_values.loc[valid]
        y_values = y_values.loc[valid]

        x_min = float(x_values.min())
        x_max = float(x_values.max())
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        if abs(x_max - x_min) < 1e-9:
            x_min -= 0.5
            x_max += 0.5
        if abs(y_max - y_min) < 1e-9:
            y_min -= 0.5
            y_max += 0.5

        x_pad = (x_max - x_min) * 0.06
        y_pad = (y_max - y_min) * 0.06
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        painter.setPen(QPen(QColor("#B0BEC5"), 1))
        painter.drawRect(plot_rect)

        for row_index, row in frame.iterrows():
            x_value = float(pd.to_numeric(row.get(self._x_column), errors="coerce"))
            y_value = float(pd.to_numeric(row.get(self._y_column), errors="coerce"))
            x_ratio = (x_value - x_min) / max(x_max - x_min, 1e-9)
            y_ratio = (y_value - y_min) / max(y_max - y_min, 1e-9)
            px = plot_rect.left() + x_ratio * plot_rect.width()
            py = plot_rect.bottom() - y_ratio * plot_rect.height()
            color = live_label_color(row.get("label", "unknown"))
            painter.setPen(QPen(color.darker(130), 1))
            painter.setBrush(QBrush(color))
            radius = 4 if row_index == frame.index[-1] else 3
            painter.drawEllipse(int(round(px - radius)), int(round(py - radius)), radius * 2, radius * 2)
            if row_index == self._selected_row_index:
                painter.setPen(QPen(QColor("#111111"), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(int(round(px - radius - 3)), int(round(py - radius - 3)), (radius + 3) * 2, (radius + 3) * 2)

        painter.setPen(QColor("#263238"))
        painter.drawText(plot_rect.left(), self.height() - 10, self._x_column)
        painter.save()
        painter.translate(14, plot_rect.bottom())
        painter.rotate(-90)
        painter.drawText(0, 0, self._y_column)
        painter.restore()


class LiveScatter3DWidget(QWidget):
    """Интерактивное 3D-облако с кликом по окнам, хвостом и октаэдром."""

    pointSelected = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self._frame = pd.DataFrame()
        self._x_column = "A"
        self._y_column = "Gamma"
        self._z_column = "V"
        self._title = "3D-карта окон"
        self._yaw_deg = -34.0
        self._pitch_deg = 22.0
        self._zoom = 0.92
        self._last_mouse_pos: tuple[float, float] | None = None
        self._press_mouse_pos: tuple[float, float] | None = None
        self._drag_distance = 0.0
        self._selected_row_index: int | None = None
        self._projected_points: list[dict[str, Any]] = []
        self._tail_length = 18
        self.setMinimumHeight(320)
        self.setMouseTracking(True)

    def set_preview(self, frame: pd.DataFrame, *, x_column: str, y_column: str, z_column: str, title: str) -> None:
        """Обновляет облако точек и выбранные оси."""

        self._frame = frame.copy() if frame is not None else pd.DataFrame()
        self._x_column = x_column
        self._y_column = y_column
        self._z_column = z_column
        if self._selected_row_index is not None and self._selected_row_index not in self._frame.index:
            self._selected_row_index = None
        self._title = title
        self.update()

    def set_selected_row_index(self, row_index: int | None) -> None:
        """Подсвечивает выбранное окно по индексу строки preview-фрейма."""

        self._selected_row_index = row_index if row_index in self._frame.index else None
        self.update()

    def reset_view(self) -> None:
        """Сбрасывает поворот и масштаб к исходным."""

        self._yaw_deg = -34.0
        self._pitch_deg = 22.0
        self._zoom = 0.92
        self.update()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        """Запоминает стартовую позицию для вращения и возможного выбора точки."""

        if event.button() == Qt.LeftButton:
            position = event.position()
            self._last_mouse_pos = (float(position.x()), float(position.y()))
            self._press_mouse_pos = self._last_mouse_pos
            self._drag_distance = 0.0
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        """Вращает 3D-сцену движением мыши."""

        if self._last_mouse_pos is not None:
            position = event.position()
            x_pos = float(position.x())
            y_pos = float(position.y())
            delta_x = x_pos - self._last_mouse_pos[0]
            delta_y = y_pos - self._last_mouse_pos[1]
            self._drag_distance += math.hypot(delta_x, delta_y)
            self._yaw_deg += delta_x * 0.55
            self._pitch_deg = min(max(self._pitch_deg - delta_y * 0.45, -82.0), 82.0)
            self._last_mouse_pos = (x_pos, y_pos)
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        """Завершает вращение или выбирает ближайшую точку при клике."""

        if event.button() == Qt.LeftButton:
            position = event.position()
            if self._press_mouse_pos is not None and self._drag_distance <= 6.0:
                self._select_nearest_point(float(position.x()), float(position.y()))
        self._last_mouse_pos = None
        self._press_mouse_pos = None
        self._drag_distance = 0.0
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:  # noqa: N802
        """Сбрасывает вид двойным кликом."""

        self.reset_view()
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: Any) -> None:  # noqa: N802
        """Позволяет менять масштаб колёсиком."""

        delta = event.angleDelta().y()
        if delta:
            zoom_step = 0.07 if delta > 0 else -0.07
            self._zoom = min(max(self._zoom + zoom_step, 0.45), 1.85)
            self.update()
        super().wheelEvent(event)

    def paintEvent(self, event: Any) -> None:  # noqa: N802
        """Рисует перспективную проекцию 3D-облака точек."""

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))

        margin_left = 20
        margin_right = 20
        margin_top = 38
        margin_bottom = 32
        plot_rect = self.rect().adjusted(margin_left, margin_top, -margin_right, -margin_bottom)
        self._projected_points = []

        painter.setPen(QPen(QColor("#263238"), 1))
        painter.drawText(12, 24, self._title)

        required_columns = (self._x_column, self._y_column, self._z_column)
        if self._frame.empty or any(column_name not in self._frame.columns for column_name in required_columns):
            painter.setPen(QColor("#546E7A"))
            painter.drawRect(plot_rect)
            painter.drawText(plot_rect, int(Qt.AlignCenter), "Жду первые 3D-точки…")
            return

        x_values = pd.to_numeric(self._frame[self._x_column], errors="coerce")
        y_values = pd.to_numeric(self._frame[self._y_column], errors="coerce")
        z_values = pd.to_numeric(self._frame[self._z_column], errors="coerce")
        valid = x_values.notna() & y_values.notna() & z_values.notna()
        if not bool(valid.any()):
            painter.setPen(QColor("#546E7A"))
            painter.drawRect(plot_rect)
            painter.drawText(plot_rect, int(Qt.AlignCenter), "Пока нет валидных 3D-координат.")
            return

        frame = self._frame.loc[valid].copy()
        x_values = x_values.loc[valid]
        y_values = y_values.loc[valid]
        z_values = z_values.loc[valid]

        x_norm, y_norm, z_norm = self._normalize_for_octahedron(x_values, y_values, z_values)

        self._draw_octahedron(painter, plot_rect)

        axis_specs = (
            ((1.1, 0.0, 0.0), self._x_column, QColor("#E53935")),
            ((0.0, 1.1, 0.0), self._y_column, QColor("#1E88E5")),
            ((0.0, 0.0, 1.1), self._z_column, QColor("#43A047")),
        )
        origin = self._project_3d_point(0.0, 0.0, 0.0, plot_rect)
        for (axis_x, axis_y, axis_z), axis_name, axis_color in axis_specs:
            end_point = self._project_3d_point(axis_x, axis_y, axis_z, plot_rect)
            painter.setPen(QPen(axis_color, 2))
            painter.drawLine(
                int(round(origin[0])),
                int(round(origin[1])),
                int(round(end_point[0])),
                int(round(end_point[1])),
            )
            painter.drawText(int(round(end_point[0] + 6)), int(round(end_point[1] - 6)), axis_name)
        painter.setPen(QColor("#607D8B"))
        painter.drawText(int(round(origin[0] + 6)), int(round(origin[1] + 14)), "C")

        ordered_points: list[dict[str, Any]] = []
        total_points = max(len(frame.index), 1)
        for order_index, (row_index, row) in enumerate(frame.iterrows()):
            px, py, depth = self._project_3d_point(
                float(x_norm.loc[row_index]),
                float(y_norm.loc[row_index]),
                float(z_norm.loc[row_index]),
                plot_rect,
            )
            color = QColor(live_label_color(row.get("label", "unknown")))
            alpha = 90 + int(round(140 * (order_index + 1) / total_points))
            color.setAlpha(min(max(alpha, 35), 255))
            radius = 5 if order_index == total_points - 1 else 3
            ordered_points.append(
                {
                    "row_index": int(row_index),
                    "screen_x": px,
                    "screen_y": py,
                    "depth": depth,
                    "color": color,
                    "radius": radius,
                }
            )

        self._draw_tail(painter, ordered_points)

        self._projected_points = ordered_points.copy()
        for point in sorted(ordered_points, key=lambda item: item["depth"]):
            color = QColor(point["color"])
            outline = QColor(color)
            outline.setAlpha(220)
            radius = int(point["radius"])
            painter.setPen(QPen(outline.darker(145), 1))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(
                int(round(point["screen_x"] - radius)),
                int(round(point["screen_y"] - radius)),
                radius * 2,
                radius * 2,
            )
            if point["row_index"] == self._selected_row_index:
                painter.setPen(QPen(QColor("#111111"), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(
                    int(round(point["screen_x"] - radius - 3)),
                    int(round(point["screen_y"] - radius - 3)),
                    (radius + 3) * 2,
                    (radius + 3) * 2,
                )

        painter.setPen(QColor("#455A64"))
        painter.drawText(
            plot_rect.adjusted(0, 0, -6, -6),
            int(Qt.AlignBottom | Qt.AlignRight),
            f"yaw {self._yaw_deg:.0f}°  pitch {self._pitch_deg:.0f}°  zoom {self._zoom:.2f}",
        )
        painter.drawText(
            plot_rect.adjusted(10, 8, -10, -8),
            int(Qt.AlignTop | Qt.AlignLeft),
            "ЛКМ: вращение/выбор  •  колесо: масштаб  •  двойной клик: сброс",
        )

    def _draw_octahedron(self, painter: QPainter, plot_rect: Any) -> None:
        """Рисует опорный октаэдр вокруг центра сцены."""

        vertices = {
            "xp": (1.0, 0.0, 0.0),
            "xn": (-1.0, 0.0, 0.0),
            "yp": (0.0, 1.0, 0.0),
            "yn": (0.0, -1.0, 0.0),
            "zp": (0.0, 0.0, 1.0),
            "zn": (0.0, 0.0, -1.0),
        }
        edges = (
            ("xp", "yp"), ("xp", "yn"), ("xp", "zp"), ("xp", "zn"),
            ("xn", "yp"), ("xn", "yn"), ("xn", "zp"), ("xn", "zn"),
            ("yp", "zp"), ("yp", "zn"), ("yn", "zp"), ("yn", "zn"),
        )
        projected = {
            key: self._project_3d_point(x_val, y_val, z_val, plot_rect)
            for key, (x_val, y_val, z_val) in vertices.items()
        }
        painter.setPen(QPen(QColor("#D6DEE6"), 1))
        for start_key, end_key in edges:
            start_point = projected[start_key]
            end_point = projected[end_key]
            painter.drawLine(
                int(round(start_point[0])),
                int(round(start_point[1])),
                int(round(end_point[0])),
                int(round(end_point[1])),
            )

    def _draw_tail(self, painter: QPainter, ordered_points: list[dict[str, Any]]) -> None:
        """Рисует хвост последовательности последних окон."""

        if len(ordered_points) < 2:
            return
        tail_points = ordered_points[-self._tail_length :]
        total_segments = max(len(tail_points) - 1, 1)
        for segment_index in range(1, len(tail_points)):
            start_point = tail_points[segment_index - 1]
            end_point = tail_points[segment_index]
            color = QColor(end_point["color"])
            alpha = 55 + int(round(170 * segment_index / total_segments))
            color.setAlpha(min(alpha, 255))
            pen_width = 1 if segment_index < len(tail_points) - 4 else 2
            painter.setPen(QPen(color, pen_width))
            painter.drawLine(
                int(round(start_point["screen_x"])),
                int(round(start_point["screen_y"])),
                int(round(end_point["screen_x"])),
                int(round(end_point["screen_y"])),
            )

    def _select_nearest_point(self, x_pos: float, y_pos: float) -> None:
        """Выбирает ближайшую к курсору точку."""

        best_point: dict[str, Any] | None = None
        best_distance = 10.5
        for point in self._projected_points:
            distance = math.hypot(point["screen_x"] - x_pos, point["screen_y"] - y_pos)
            if distance <= best_distance:
                best_point = point
                best_distance = distance
        if best_point is None:
            return
        self._selected_row_index = int(best_point["row_index"])
        self.pointSelected.emit(self._selected_row_index)
        self.update()

    def _normalize_signed_axis(self, values: pd.Series) -> pd.Series:
        """Переносит одну ось в симметричный диапазон [-1, 1]."""

        min_value = float(values.min())
        max_value = float(values.max())
        if abs(max_value - min_value) < 1e-9:
            return pd.Series([0.0] * len(values), index=values.index, dtype=float)
        center = (min_value + max_value) * 0.5
        half_range = max(abs(max_value - center), abs(min_value - center), 1e-9)
        return (values - center) / half_range

    def _normalize_for_octahedron(
        self,
        x_values: pd.Series,
        y_values: pd.Series,
        z_values: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Масштабирует облако так, чтобы все точки лежали внутри октаэдра |x|+|y|+|z|<=1."""

        x_norm = self._normalize_signed_axis(x_values)
        y_norm = self._normalize_signed_axis(y_values)
        z_norm = self._normalize_signed_axis(z_values)
        l1_radius = x_norm.abs() + y_norm.abs() + z_norm.abs()
        scale = max(float(l1_radius.max()), 1.0)
        return x_norm / scale, y_norm / scale, z_norm / scale

    def _project_3d_point(self, x_value: float, y_value: float, z_value: float, plot_rect: Any) -> tuple[float, float, float]:
        """Преобразует нормализованную 3D-точку в экранную перспективу."""

        yaw = math.radians(self._yaw_deg)
        pitch = math.radians(self._pitch_deg)

        x_yaw = x_value * math.cos(yaw) + z_value * math.sin(yaw)
        z_yaw = -x_value * math.sin(yaw) + z_value * math.cos(yaw)

        y_pitch = y_value * math.cos(pitch) - z_yaw * math.sin(pitch)
        z_pitch = y_value * math.sin(pitch) + z_yaw * math.cos(pitch)

        camera_distance = 4.2
        perspective = camera_distance / max(camera_distance - z_pitch, 0.6)
        scale = min(plot_rect.width(), plot_rect.height()) * 0.23 * self._zoom
        screen_x = plot_rect.center().x() + x_yaw * scale * perspective
        screen_y = plot_rect.center().y() - y_pitch * scale * perspective
        return float(screen_x), float(screen_y), float(z_pitch)


class Live3DFullscreenWindow(QWidget):
    """Отдельное полноэкранное окно для 3D-предпросмотра."""

    closed = Signal()
    pointSelected = Signal(int)

    def __init__(self) -> None:
        super().__init__(None, Qt.Window)
        self.setWindowTitle("ИИС · 3D просмотр")
        self.resize(1440, 900)

        layout = QVBoxLayout(self)
        top_bar = QHBoxLayout()
        self.title_label = QLabel("3D-предпросмотр")
        self.status_label = QLabel("Esc: выход из fullscreen")
        self.reset_button = QPushButton("Сбросить вид")
        self.close_button = QPushButton("Закрыть")

        top_bar.addWidget(self.title_label, stretch=1)
        top_bar.addWidget(self.status_label, stretch=0)
        top_bar.addWidget(self.reset_button, stretch=0)
        top_bar.addWidget(self.close_button, stretch=0)

        self.plot_widget = LiveScatter3DWidget()
        self.detail_browser = QTextBrowser()
        self.detail_browser.setMaximumHeight(220)
        self.detail_browser.setHtml(
            "<h4>Детали окна</h4>"
            "<p>Щёлкните по точке, чтобы увидеть состав окна.</p>"
        )

        layout.addLayout(top_bar)
        layout.addWidget(self.plot_widget, stretch=1)
        layout.addWidget(self.detail_browser, stretch=0)

        self.reset_button.clicked.connect(self.plot_widget.reset_view)
        self.close_button.clicked.connect(self.close)
        self.plot_widget.pointSelected.connect(self.pointSelected.emit)

    def set_preview(self, frame: pd.DataFrame, *, x_column: str, y_column: str, z_column: str, title: str) -> None:
        """Обновляет полноэкранное 3D-окно текущими данными."""

        self.title_label.setText(title)
        self.plot_widget.set_preview(frame, x_column=x_column, y_column=y_column, z_column=z_column, title=title)

    def set_selected_row_index(self, row_index: int | None) -> None:
        """Подсвечивает выбранное окно."""

        self.plot_widget.set_selected_row_index(row_index)

    def set_detail_html(self, html: str) -> None:
        """Обновляет карточку выбранного окна."""

        self.detail_browser.setHtml(html)

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        """Позволяет закрыть fullscreen через Escape."""

        if event.key() == int(Qt.Key_Escape):
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: Any) -> None:  # noqa: N802
        """Уведомляет основное окно о закрытии fullscreen-режима."""

        self.closed.emit()
        super().closeEvent(event)


class GuiLogHandler(logging.Handler):
    """Переадресует сообщения logging в Qt-сигнал."""

    def __init__(self, callback: Any) -> None:
        super().__init__()
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        """Передаёт отформатированное сообщение в GUI."""

        try:
            self._callback(self.format(record))
        except Exception:
            self.handleError(record)


class AnalysisWorker(QObject):
    """Фоновый исполнитель анализа, чтобы GUI не блокировался."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    progress_state = Signal(object)

    def __init__(
        self,
        dataset: str,
        mode: str,
        dataset_path: Path,
        dynamic: bool,
        window_seconds: float,
        step_seconds: float,
        focus_version: str,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.mode = mode
        self.dataset_path = dataset_path
        self.dynamic = dynamic
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.focus_version = focus_version

    def run(self) -> None:
        """Запускает анализ и возвращает сводку."""

        run_label = f"{self.dataset}_{self.mode}_{'dynamic' if self.dynamic else 'static'}_{self.focus_version}"
        configure_logging("INFO", run_label=run_label, use_stdout=False)
        log_handler = GuiLogHandler(self.progress.emit)
        log_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)

        try:
            def progress_callback(payload: dict[str, Any]) -> None:
                self.progress_state.emit(payload)

            self.progress.emit(
                f"Старт анализа: dataset={self.dataset}, mode={self.mode}, dynamic={self.dynamic}, path={self.dataset_path}"
            )
            summary = run_dataset(
                dataset=self.dataset,
                mode=self.mode,
                dataset_path=self.dataset_path,
                dynamic=self.dynamic,
                window_seconds=self.window_seconds if self.dynamic else None,
                step_seconds=self.step_seconds if self.dynamic else None,
                focus_version=self.focus_version,
                progress_callback=progress_callback,
            )
            self.finished.emit(
                {
                    "dataset": self.dataset,
                    "mode": self.mode,
                    "dynamic": self.dynamic,
                    "window_seconds": self.window_seconds,
                    "step_seconds": self.step_seconds,
                    "focus_version": self.focus_version,
                    "summary": summary,
                }
            )
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            root_logger.removeHandler(log_handler)
            log_handler.close()


class MainWindow(QMainWindow):
    """Главное окно интерфейса анализа ИИС."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ИИС Аналитика")
        self.resize(1600, 980)

        self.worker_thread: QThread | None = None
        self.worker: AnalysisWorker | None = None

        self.features_df = pd.DataFrame()
        self.results_df = pd.DataFrame()
        self.comparison_df = pd.DataFrame()
        self.dynamic_df = pd.DataFrame()
        self.live_preview_df = pd.DataFrame()
        self.current_summary: dict[str, Any] = {}
        self.current_payload: dict[str, Any] = {}
        self.segment_index_map: dict[str, int] = {}
        self.analysis_started_at = 0.0
        self.intervention_df = pd.DataFrame()
        self._last_progress_stage = ""
        self._last_progress_message = ""
        self._last_progress_log_at = 0.0
        self._live_preview_version = ""
        self._live_preview_count = 0
        self._live_preview_total = 0
        self._live_axis_syncing = False
        self._live_selected_row_index: int | None = None
        self.live_fullscreen_window: Live3DFullscreenWindow | None = None

        self.features_model = DataFrameTableModel()
        self.versions_model = DataFrameTableModel()
        self.comparison_model = DataFrameTableModel()
        self.dynamic_model = DataFrameTableModel()
        self.live_preview_model = DataFrameTableModel()
        self.intervention_model = DataFrameTableModel()

        self._build_ui()
        self._apply_initial_defaults()

    def _build_ui(self) -> None:
        """Собирает интерфейс окна."""

        root_widget = QWidget()
        self.setCentralWidget(root_widget)
        self.setStatusBar(QStatusBar())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_controls_panel())
        splitter.addWidget(self._build_tabs_panel())
        splitter.setSizes([380, 1220])

        layout = QVBoxLayout(root_widget)
        layout.addWidget(splitter)

    def _build_controls_panel(self) -> QWidget:
        """Создаёт левую панель управления."""

        widget = QWidget()
        layout = QVBoxLayout(widget)

        scenario_group = QGroupBox("Параметры сценария")
        scenario_form = QFormLayout(scenario_group)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(SUPPORTED_DATASETS)
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_changed)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(PROCESSING_MODES)
        self.mode_combo.currentTextChanged.connect(lambda _: self._refresh_scenario_help())

        self.dynamic_checkbox = QCheckBox("Включить динамический сценарий")
        self.dynamic_checkbox.setChecked(True)
        self.dynamic_checkbox.toggled.connect(self._update_dynamic_controls)

        self.focus_version_combo = QComboBox()
        self.focus_version_combo.addItems(MODEL_ORDER)
        self.focus_version_combo.setCurrentText("IISVersion6")
        self.focus_version_combo.currentTextChanged.connect(lambda _: self._refresh_scenario_help())

        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(1.0, 180.0)
        self.window_spin.setDecimals(1)
        self.window_spin.setSingleStep(1.0)
        self.window_spin.setSuffix(" c")
        self.window_spin.valueChanged.connect(lambda _: self._refresh_scenario_help())

        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.5, 120.0)
        self.step_spin.setDecimals(1)
        self.step_spin.setSingleStep(0.5)
        self.step_spin.setSuffix(" c")
        self.step_spin.valueChanged.connect(lambda _: self._refresh_scenario_help())

        self.path_edit = QLineEdit()
        self.path_edit.textChanged.connect(lambda _: self._refresh_scenario_help())
        browse_button = QPushButton("Выбрать…")
        browse_button.clicked.connect(self._browse_dataset_path)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.path_edit, stretch=1)
        path_layout.addWidget(browse_button)

        scenario_form.addRow("Датасет", self.dataset_combo)
        scenario_form.addRow("Режим", self.mode_combo)
        scenario_form.addRow("", self.dynamic_checkbox)
        scenario_form.addRow("Версия для динамики", self.focus_version_combo)
        scenario_form.addRow("Размер окна", self.window_spin)
        scenario_form.addRow("Шаг окна", self.step_spin)
        scenario_form.addRow("Путь к датасету", path_layout)

        actions_group = QGroupBox("Запуск")
        actions_layout = QVBoxLayout(actions_group)
        self.run_button = QPushButton("Запустить анализ")
        self.run_button.clicked.connect(self._start_analysis)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_stage_label = QLabel("Ожидание запуска.")
        self.progress_eta_label = QLabel("ETA: --")
        open_outputs_button = QPushButton("Открыть outputs")
        open_outputs_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(OUTPUT_DIR))))
        actions_layout.addWidget(self.run_button)
        actions_layout.addWidget(self.progress_bar)
        actions_layout.addWidget(self.progress_stage_label)
        actions_layout.addWidget(self.progress_eta_label)
        actions_layout.addWidget(open_outputs_button)

        help_group = QGroupBox("Пояснение сценария")
        help_layout = QVBoxLayout(help_group)
        self.scenario_help = QTextBrowser()
        help_layout.addWidget(self.scenario_help)

        layout.addWidget(scenario_group)
        layout.addWidget(actions_group)
        layout.addWidget(help_group, stretch=1)
        return widget

    def _build_tabs_panel(self) -> QWidget:
        """Создаёт вкладки результатов."""

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.tabs = QTabWidget()
        self.summary_tab = self._build_summary_tab()
        self.live_tab = self._build_live_tab()
        self.segment_tab = self._build_segment_tab()
        self.dynamic_tab = self._build_dynamic_tab()
        self.intervention_tab = self._build_intervention_tab()
        self.log_tab = self._build_log_tab()
        self.tabs.addTab(self.summary_tab, "Сводка")
        self.tabs.addTab(self.live_tab, "Live")
        self.tabs.addTab(self.segment_tab, "Сегмент")
        self.tabs.addTab(self.dynamic_tab, "Динамика")
        self.tabs.addTab(self.intervention_tab, "Интервенция")
        self.tabs.addTab(self.log_tab, "Лог")
        layout.addWidget(self.tabs)
        return widget

    def _build_summary_tab(self) -> QWidget:
        """Создаёт вкладку сводки и графиков."""

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.summary_browser = QTextBrowser()
        self.comparison_table = QTableView()
        self._configure_table(self.comparison_table, self.comparison_model)

        plot_controls = QHBoxLayout()
        self.static_plot_combo = QComboBox()
        self.static_plot_combo.currentIndexChanged.connect(self._update_static_plot)
        plot_controls.addWidget(QLabel("График"))
        plot_controls.addWidget(self.static_plot_combo, stretch=1)

        self.static_plot_viewer = PlotViewer("Сводные графики по выбранному запуску")
        layout.addWidget(self.summary_browser, stretch=0)
        layout.addWidget(self.comparison_table, stretch=1)
        layout.addLayout(plot_controls)
        layout.addWidget(self.static_plot_viewer, stretch=2)
        return widget

    def _build_live_tab(self) -> QWidget:
        """Создаёт вкладку живого предпросмотра точек во время расчёта."""

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.live_note_browser = QTextBrowser()
        self.live_note_browser.setHtml(
            "<h3>Живой предпросмотр</h3>"
            "Во время расчёта выбранной версии здесь появляются уже посчитанные окна. "
            "Это промежуточная карта, поэтому координаты и пороги могут немного меняться до завершения запуска."
        )
        axes_layout = QHBoxLayout()
        self.live_axis_x_combo = QComboBox()
        self.live_axis_y_combo = QComboBox()
        self.live_axis_z_combo = QComboBox()
        for combo in (self.live_axis_x_combo, self.live_axis_y_combo, self.live_axis_z_combo):
            combo.currentTextChanged.connect(self._on_live_axes_changed)
        axes_layout.addWidget(QLabel("X"))
        axes_layout.addWidget(self.live_axis_x_combo, stretch=1)
        axes_layout.addWidget(QLabel("Y"))
        axes_layout.addWidget(self.live_axis_y_combo, stretch=1)
        axes_layout.addWidget(QLabel("Z"))
        axes_layout.addWidget(self.live_axis_z_combo, stretch=1)
        self.live_reset_3d_button = QPushButton("Сбросить 3D")
        self.live_reset_3d_button.clicked.connect(lambda: self.live_scatter_3d_widget.reset_view())
        self.live_fullscreen_button = QPushButton("3D на весь экран")
        self.live_fullscreen_button.clicked.connect(self._open_live_3d_fullscreen)
        axes_layout.addWidget(self.live_reset_3d_button, stretch=0)
        axes_layout.addWidget(self.live_fullscreen_button, stretch=0)
        self.live_scatter_widget = LiveScatterWidget()
        self.live_scatter_3d_widget = LiveScatter3DWidget()
        self.live_scatter_3d_widget.pointSelected.connect(self._on_live_point_selected)
        self.live_detail_browser = QTextBrowser()
        self.live_detail_browser.setMaximumHeight(180)
        self.live_detail_browser.setHtml(
            "<h4>Детали окна</h4>"
            "<p>Щёлкните по точке на 3D-карте или по строке в таблице, чтобы увидеть сведения об окне.</p>"
        )
        self.live_table = QTableView()
        self._configure_table(self.live_table, self.live_preview_model)
        self.live_table.clicked.connect(self._on_live_table_clicked)
        preview_splitter = QSplitter(Qt.Horizontal)
        preview_splitter.addWidget(self.live_scatter_widget)
        preview_splitter.addWidget(self.live_scatter_3d_widget)
        preview_splitter.setSizes([560, 560])

        layout.addWidget(self.live_note_browser, stretch=0)
        layout.addLayout(axes_layout)
        layout.addWidget(preview_splitter, stretch=2)
        layout.addWidget(self.live_detail_browser, stretch=0)
        layout.addWidget(self.live_table, stretch=1)
        return widget

    def _build_segment_tab(self) -> QWidget:
        """Создаёт вкладку по выбранному сегменту."""

        widget = QWidget()
        layout = QVBoxLayout(widget)

        segment_controls = QHBoxLayout()
        self.segment_combo = QComboBox()
        self.segment_combo.setEditable(True)
        self.segment_combo.currentTextChanged.connect(self._update_segment_views)
        segment_controls.addWidget(QLabel("Сегмент"))
        segment_controls.addWidget(self.segment_combo, stretch=1)

        tables_layout = QGridLayout()
        self.features_table = QTableView()
        self._configure_table(self.features_table, self.features_model)
        self.versions_table = QTableView()
        self._configure_table(self.versions_table, self.versions_model)
        tables_layout.addWidget(QLabel("Входные признаки"), 0, 0)
        tables_layout.addWidget(QLabel("Версии модели"), 0, 1)
        tables_layout.addWidget(self.features_table, 1, 0)
        tables_layout.addWidget(self.versions_table, 1, 1)

        self.segment_note_browser = QTextBrowser()

        layout.addLayout(segment_controls)
        layout.addLayout(tables_layout, stretch=1)
        layout.addWidget(self.segment_note_browser, stretch=0)
        return widget

    def _build_dynamic_tab(self) -> QWidget:
        """Создаёт вкладку динамического анализа."""

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.dynamic_note_browser = QTextBrowser()

        record_controls = QHBoxLayout()
        self.dynamic_record_combo = QComboBox()
        self.dynamic_record_combo.currentTextChanged.connect(self._update_dynamic_table)
        record_controls.addWidget(QLabel("Запись"))
        record_controls.addWidget(self.dynamic_record_combo, stretch=1)

        self.dynamic_table = QTableView()
        self._configure_table(self.dynamic_table, self.dynamic_model)
        self.dynamic_plot_viewer = PlotViewer("Причинная динамика IIS")

        layout.addWidget(self.dynamic_note_browser, stretch=0)
        layout.addLayout(record_controls)
        layout.addWidget(self.dynamic_table, stretch=1)
        layout.addWidget(self.dynamic_plot_viewer, stretch=2)
        return widget

    def _build_intervention_tab(self) -> QWidget:
        """Создаёт вкладку псевдо-причинных интервенций."""

        widget = QWidget()
        layout = QVBoxLayout(widget)

        controls = QGridLayout()
        self.intervention_record_combo = QComboBox()
        self.intervention_record_combo.currentTextChanged.connect(self._sync_intervention_time_range)
        self.intervention_target_combo = QComboBox()
        self.intervention_target_combo.addItems(INTERVENTION_TARGETS)
        self.intervention_operation_combo = QComboBox()
        self.intervention_operation_combo.addItems(("add", "scale"))
        self.intervention_start_spin = QDoubleSpinBox()
        self.intervention_start_spin.setRange(0.0, 100000.0)
        self.intervention_start_spin.setDecimals(1)
        self.intervention_start_spin.setSuffix(" c")
        self.intervention_end_spin = QDoubleSpinBox()
        self.intervention_end_spin.setRange(0.0, 100000.0)
        self.intervention_end_spin.setDecimals(1)
        self.intervention_end_spin.setSuffix(" c")
        self.intervention_magnitude_spin = QDoubleSpinBox()
        self.intervention_magnitude_spin.setRange(-100000.0, 100000.0)
        self.intervention_magnitude_spin.setDecimals(4)
        self.intervention_magnitude_spin.setValue(0.15)

        apply_button = QPushButton("Симулировать интервенцию")
        apply_button.clicked.connect(self._run_intervention)

        controls.addWidget(QLabel("Запись"), 0, 0)
        controls.addWidget(self.intervention_record_combo, 0, 1)
        controls.addWidget(QLabel("Признак"), 1, 0)
        controls.addWidget(self.intervention_target_combo, 1, 1)
        controls.addWidget(QLabel("Операция"), 2, 0)
        controls.addWidget(self.intervention_operation_combo, 2, 1)
        controls.addWidget(QLabel("Начало"), 3, 0)
        controls.addWidget(self.intervention_start_spin, 3, 1)
        controls.addWidget(QLabel("Конец"), 4, 0)
        controls.addWidget(self.intervention_end_spin, 4, 1)
        controls.addWidget(QLabel("Амплитуда"), 5, 0)
        controls.addWidget(self.intervention_magnitude_spin, 5, 1)
        controls.addWidget(apply_button, 6, 0, 1, 2)

        self.intervention_note_browser = QTextBrowser()
        self.intervention_note_browser.setHtml(
            "Интервенции здесь псевдо-причинные: меняется извлечённый признак окна, затем заново пересчитываются IIS и IIS_dynamic. "
            "Это стресс-тест модели, а не имитация прямого биомедицинского измерения."
        )
        self.intervention_table = QTableView()
        self._configure_table(self.intervention_table, self.intervention_model)
        self.intervention_plot_viewer = PlotViewer("Интервенционный сценарий")

        layout.addLayout(controls)
        layout.addWidget(self.intervention_note_browser, stretch=0)
        layout.addWidget(self.intervention_table, stretch=1)
        layout.addWidget(self.intervention_plot_viewer, stretch=2)
        return widget

    def _build_log_tab(self) -> QWidget:
        """Создаёт вкладку лога выполнения."""

        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)
        return widget

    def _configure_table(self, table: QTableView, model: DataFrameTableModel) -> None:
        """Применяет общие настройки таблиц."""

        table.setModel(model)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableView.SelectRows)
        table.setSelectionMode(QTableView.SingleSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)

    def _apply_initial_defaults(self) -> None:
        """Задаёт значения по умолчанию после инициализации окна."""

        if "ds002724" in SUPPORTED_DATASETS:
            self.dataset_combo.setCurrentText("ds002724")
        self.mode_combo.setCurrentText("strict")
        self._on_dataset_changed(self.dataset_combo.currentText())
        self._update_dynamic_controls(self.dynamic_checkbox.isChecked())
        self.summary_browser.setText("Запустите анализ, чтобы загрузить сводку.")
        self.live_note_browser.setText("Запустите анализ, чтобы увидеть, как окна превращаются в точки состояния.")
        self.live_preview_model.set_frame(pd.DataFrame())
        self.live_scatter_widget.set_preview(pd.DataFrame(), x_column="IIS", y_column="RES", title="Живое облако состояний")
        self.live_scatter_3d_widget.set_preview(pd.DataFrame(), x_column="A", y_column="Gamma", z_column="V", title="Живая 3D-карта")
        self.live_detail_browser.setHtml(
            "<h4>Детали окна</h4>"
            "<p>Щёлкните по точке на 3D-карте или по строке в таблице, чтобы увидеть сведения об окне.</p>"
        )
        self._set_live_axis_options([], preferred_axes=("A", "Gamma", "V"))
        self.segment_note_browser.setText("Выберите сценарий и запустите расчёт.")
        self.dynamic_note_browser.setText("Динамическая вкладка заполнится после запуска анализа в режиме динамики.")
        self.progress_stage_label.setText("Ожидание запуска.")
        self.progress_eta_label.setText("ETA: --")

    def _on_dataset_changed(self, dataset: str) -> None:
        """Подставляет путь и рекомендуемые окна при смене датасета."""

        self.path_edit.setText(str(Path(DATASET_DEFAULTS[dataset]["default_root"]).resolve()))
        dynamic_defaults = DYNAMIC_WINDOW_DEFAULTS.get(dataset, {"window_seconds": 5.0, "step_seconds": 2.5})
        self.window_spin.setValue(float(dynamic_defaults["window_seconds"]))
        self.step_spin.setValue(float(dynamic_defaults["step_seconds"]))
        self._refresh_scenario_help()

    def _update_dynamic_controls(self, enabled: bool) -> None:
        """Включает или выключает элементы динамического сценария."""

        self.focus_version_combo.setEnabled(enabled)
        self.window_spin.setEnabled(enabled)
        self.step_spin.setEnabled(enabled)
        self._refresh_scenario_help()

    def _refresh_scenario_help(self) -> None:
        """Обновляет краткое пояснение выбранного сценария."""

        lines = [
            f"<b>Датасет:</b> {self.dataset_combo.currentText()}",
            f"<b>Режим пропусков:</b> {self.mode_combo.currentText()}",
            f"<b>Путь:</b> {self.path_edit.text()}",
        ]
        if self.dynamic_checkbox.isChecked():
            lines.extend(
                [
                    "<b>Динамический сценарий:</b> включён",
                    f"<b>Версия для динамики:</b> {self.focus_version_combo.currentText()}",
                    f"<b>Окно:</b> {self.window_spin.value():.1f} с",
                    f"<b>Шаг:</b> {self.step_spin.value():.1f} с",
                    "Причинное сглаживание использует только прошлые окна и сохраняет отдельную траекторию IIS_dynamic.",
                ]
            )
        else:
            lines.append("<b>Динамический сценарий:</b> выключен. Будут показаны только статические результаты.")
        self.scenario_help.setHtml("<br>".join(lines))

    def _browse_dataset_path(self) -> None:
        """Даёт выбрать каталог датасета вручную."""

        selected_dir = QFileDialog.getExistingDirectory(self, "Выберите каталог датасета", self.path_edit.text())
        if selected_dir:
            self.path_edit.setText(selected_dir)

    def _start_analysis(self) -> None:
        """Запускает анализ в отдельном потоке."""

        dataset_path = Path(self.path_edit.text()).expanduser()
        if not dataset_path.exists():
            QMessageBox.warning(self, "Нет каталога", f"Каталог не найден:\n{dataset_path}")
            return

        self.run_button.setEnabled(False)
        self.log_view.clear()
        self.analysis_started_at = time.perf_counter()
        self.progress_bar.setValue(0)
        self.progress_stage_label.setText("Подготовка запуска…")
        self.progress_eta_label.setText("ETA: --")
        self._last_progress_stage = ""
        self._last_progress_message = ""
        self._last_progress_log_at = 0.0
        self.live_preview_df = pd.DataFrame()
        self.live_preview_model.set_frame(pd.DataFrame())
        self.live_scatter_widget.set_preview(pd.DataFrame(), x_column="IIS", y_column="RES", title="Живое облако состояний")
        self.live_scatter_3d_widget.set_preview(pd.DataFrame(), x_column="A", y_column="Gamma", z_column="V", title="Живая 3D-карта")
        self.live_scatter_3d_widget.reset_view()
        self._live_preview_version = ""
        self._live_preview_count = 0
        self._live_preview_total = 0
        self._live_selected_row_index = None
        self.live_detail_browser.setHtml(
            "<h4>Детали окна</h4>"
            "<p>Жду первые окна. После появления точек можно щёлкнуть по ним и посмотреть состав окна.</p>"
        )
        self._set_live_axis_options([], preferred_axes=("A", "Gamma", "V"))
        self.live_note_browser.setHtml(
            "<h3>Живой предпросмотр</h3>"
            "Жду первые точки выбранной версии. Они появятся после извлечения признаков, на этапе расчёта модели."
        )
        self.statusBar().showMessage("Анализ запущен…")
        self.tabs.setCurrentWidget(self.live_tab)

        self.worker_thread = QThread(self)
        self.worker = AnalysisWorker(
            dataset=self.dataset_combo.currentText(),
            mode=self.mode_combo.currentText(),
            dataset_path=dataset_path,
            dynamic=self.dynamic_checkbox.isChecked(),
            window_seconds=float(self.window_spin.value()),
            step_seconds=float(self.step_spin.value()),
            focus_version=self.focus_version_combo.currentText(),
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._append_log)
        self.worker.progress_state.connect(self._update_progress_status)
        self.worker.finished.connect(self._on_analysis_finished)
        self.worker.failed.connect(self._on_analysis_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._cleanup_worker)
        self.worker_thread.start()

    def _append_log(self, message: str) -> None:
        """Добавляет строку в окно лога."""

        self.log_view.appendPlainText(message)

    def _set_live_axis_options(self, options: list[str], *, preferred_axes: tuple[str, str, str]) -> None:
        """Заполняет селекторы осей и по возможности сохраняет выбор."""

        self._live_axis_syncing = True
        try:
            for combo, preferred in zip(
                (self.live_axis_x_combo, self.live_axis_y_combo, self.live_axis_z_combo),
                preferred_axes,
                strict=False,
            ):
                current_value = combo.currentText()
                combo.blockSignals(True)
                combo.clear()
                if options:
                    combo.addItems(options)
                    if preferred in options:
                        combo.setCurrentText(preferred)
                    elif current_value in options:
                        combo.setCurrentText(current_value)
                    else:
                        combo.setCurrentIndex(0)
                combo.blockSignals(False)
        finally:
            self._live_axis_syncing = False

    def _numeric_live_columns(self, frame: pd.DataFrame) -> list[str]:
        """Возвращает доступные числовые колонки для 2D/3D-предпросмотра."""

        ordered_candidates = ("A", "Gamma", "V", "Q", "IIS", "RES", "H", "K", *frame.columns)
        numeric_columns: list[str] = []
        seen: set[str] = set()
        for column_name in ordered_candidates:
            if column_name in seen or column_name not in frame.columns:
                continue
            series = pd.to_numeric(frame[column_name], errors="coerce")
            if bool(series.notna().any()):
                numeric_columns.append(column_name)
                seen.add(column_name)
        return numeric_columns

    def _pick_live_axis(self, candidates: list[str], preferred_names: tuple[str, ...], *, fallback: str = "") -> str:
        """Выбирает лучшую доступную ось из приоритетного списка."""

        for column_name in preferred_names:
            if column_name in candidates:
                return column_name
        return fallback or (candidates[0] if candidates else "")

    def _refresh_live_views(self) -> None:
        """Обновляет 2D/3D-визуализации и пояснение к ним."""

        frame = self.live_preview_df.copy() if self.live_preview_df is not None else pd.DataFrame()
        x_column = self.live_axis_x_combo.currentText() or "IIS"
        y_column = self.live_axis_y_combo.currentText() or "RES"
        z_column = self.live_axis_z_combo.currentText() or "V"

        self.live_scatter_widget.set_preview(
            frame,
            x_column=x_column,
            y_column=y_column,
            title=f"2D: {x_column} × {y_column}",
        )
        self.live_scatter_widget.set_selected_row_index(self._live_selected_row_index)
        self.live_scatter_3d_widget.set_preview(
            frame,
            x_column=x_column,
            y_column=y_column,
            z_column=z_column,
            title=f"3D: {x_column} × {y_column} × {z_column}",
        )
        self.live_scatter_3d_widget.set_selected_row_index(self._live_selected_row_index)
        if self.live_fullscreen_window is not None:
            self.live_fullscreen_window.set_preview(
                frame,
                x_column=x_column,
                y_column=y_column,
                z_column=z_column,
                title=f"3D fullscreen: {x_column} × {y_column} × {z_column}",
            )
            self.live_fullscreen_window.set_selected_row_index(self._live_selected_row_index)

        if self._live_preview_version:
            self.live_note_browser.setHtml(
                "<h3>Живой предпросмотр</h3>"
                f"<p>Версия: <b>{self._live_preview_version}</b></p>"
                f"<p>Уже посчитано окон: <b>{self._live_preview_count}</b> из <b>{self._live_preview_total}</b>.</p>"
                f"<p>2D-оси: <b>{x_column}</b> и <b>{y_column}</b>. 3D-оси: <b>{x_column}</b>, <b>{y_column}</b>, <b>{z_column}</b>.</p>"
                "<p>3D-предпросмотр можно вращать мышью, приближать колесом и сбросить двойным кликом. "
                "Это промежуточная карта: координаты ещё могут немного сдвигаться до завершения анализа.</p>"
            )

    def _on_live_axes_changed(self) -> None:
        """Перерисовывает live-визуализации при смене осей."""

        if self._live_axis_syncing:
            return
        self._refresh_live_views()

    def _format_live_row_details(self, row: pd.Series) -> str:
        """Строит HTML-карточку по выбранному окну."""

        summary_fields = [
            ("source_record_id", "Источник"),
            ("window_start_sec", "Начало окна, с"),
            ("window_end_sec", "Конец окна, с"),
            ("label", "Метка"),
            ("state_map_4", "Квадрант"),
            ("IIS", "IIS"),
            ("RES", "RES"),
            ("A", "A"),
            ("Gamma", "Gamma"),
            ("V", "V"),
            ("Q", "Q"),
            ("H", "H"),
            ("K", "K"),
        ]
        rows_html: list[str] = []
        for column_name, title in summary_fields:
            if column_name not in row.index:
                continue
            value = row.get(column_name)
            if pd.isna(value) or value == "":
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rendered = format_float(value)
            else:
                rendered = str(value)
            rows_html.append(f"<tr><td><b>{title}</b></td><td>{rendered}</td></tr>")
        segment_title = build_segment_key(row)
        return (
            "<h4>Детали окна</h4>"
            f"<p><b>{segment_title}</b></p>"
            "<table cellspacing='6'>"
            f"{''.join(rows_html)}"
            "</table>"
        )

    def _show_live_point_details(self, row_index: int, *, focus_table: bool) -> None:
        """Показывает детали выбранного окна и синхронизирует таблицу/3D."""

        if self.live_preview_df.empty or row_index < 0 or row_index >= len(self.live_preview_df.index):
            return
        self._live_selected_row_index = int(row_index)
        row = self.live_preview_df.iloc[row_index].copy()
        self.live_scatter_widget.set_selected_row_index(self._live_selected_row_index)
        self.live_scatter_3d_widget.set_selected_row_index(self._live_selected_row_index)
        detail_html = self._format_live_row_details(row)
        self.live_detail_browser.setHtml(detail_html)
        if self.live_fullscreen_window is not None:
            self.live_fullscreen_window.set_selected_row_index(self._live_selected_row_index)
            self.live_fullscreen_window.set_detail_html(detail_html)
        if focus_table:
            self.live_table.selectRow(self._live_selected_row_index)
            model_index = self.live_preview_model.index(self._live_selected_row_index, 0)
            if model_index.isValid():
                self.live_table.scrollTo(model_index)

    def _on_live_point_selected(self, row_index: int) -> None:
        """Реагирует на клик по точке в 3D-предпросмотре."""

        self._show_live_point_details(int(row_index), focus_table=True)

    def _on_live_table_clicked(self, index: QModelIndex) -> None:
        """Показывает детали по клику в таблице live-preview."""

        if not index.isValid():
            return
        self._show_live_point_details(index.row(), focus_table=False)

    def _on_live_fullscreen_closed(self) -> None:
        """Снимает ссылку на полноэкранное окно после его закрытия."""

        self.live_fullscreen_window = None

    def _open_live_3d_fullscreen(self) -> None:
        """Открывает отдельное полноэкранное окно с 3D-визуализацией."""

        if self.live_fullscreen_window is None:
            self.live_fullscreen_window = Live3DFullscreenWindow()
            self.live_fullscreen_window.closed.connect(self._on_live_fullscreen_closed)
            self.live_fullscreen_window.pointSelected.connect(self._on_live_point_selected)

        self._refresh_live_views()
        if self._live_selected_row_index is not None and not self.live_preview_df.empty:
            self._show_live_point_details(self._live_selected_row_index, focus_table=False)
        self.live_fullscreen_window.showFullScreen()
        self.live_fullscreen_window.raise_()
        self.live_fullscreen_window.activateWindow()

    def _update_live_preview(self, payload: dict[str, Any]) -> None:
        """Обновляет вкладку живого предпросмотра по промежуточным данным модели."""

        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            return

        frame = pd.DataFrame.from_records(rows)
        preferred_columns = [
            column_name
            for column_name in ("source_record_id", "version", *LIVE_PREVIEW_COLUMNS)
            if column_name in frame.columns
        ]
        extra_columns = [column_name for column_name in frame.columns if column_name not in preferred_columns]
        frame = frame[preferred_columns + extra_columns].copy()

        self.live_preview_df = frame
        self.live_preview_model.set_frame(frame)

        numeric_columns = self._numeric_live_columns(frame)
        x_default = self._pick_live_axis(
            numeric_columns,
            (str(payload.get("x_column", "") or ""), "IIS", "A", "Q", "Gamma", "V", "RES"),
            fallback="IIS",
        )
        y_default = self._pick_live_axis(
            numeric_columns,
            (str(payload.get("y_column", "") or ""), "RES", "Q", "V", "Gamma", "A", "IIS"),
            fallback="RES",
        )
        z_default = self._pick_live_axis(
            [column_name for column_name in numeric_columns if column_name not in (x_default, y_default)],
            (str(payload.get("z_column", "") or ""), "V", "Q", "Gamma", "RES", "A", "IIS"),
            fallback="V",
        )
        self._set_live_axis_options(numeric_columns, preferred_axes=(x_default, y_default, z_default))
        self._live_preview_version = str(payload.get("version", "") or "")
        count = int(payload.get("count", len(frame)) or len(frame))
        total = int(payload.get("total", count) or count)
        self._live_preview_count = count
        self._live_preview_total = total
        self._refresh_live_views()
        selected_row_index = self._live_selected_row_index
        if selected_row_index is None or selected_row_index >= len(frame.index):
            selected_row_index = len(frame.index) - 1
        if selected_row_index >= 0:
            self._show_live_point_details(int(selected_row_index), focus_table=True)

    def _update_progress_status(self, payload: dict[str, Any]) -> None:
        """Обновляет progress bar и оценку оставшегося времени."""

        stage = str(payload.get("stage", "") or "")
        message = str(payload.get("message", "Идёт обработка") or "Идёт обработка")
        fraction = float(payload.get("fraction", 0.0) or 0.0)
        fraction = min(max(fraction, 0.0), 1.0)
        self.progress_bar.setValue(int(round(fraction * 1000)))
        self.progress_stage_label.setText(message)

        now = time.perf_counter()
        should_log = False
        if stage != self._last_progress_stage:
            should_log = True
        elif message != self._last_progress_message and now - self._last_progress_log_at >= 10.0:
            should_log = True
        elif now - self._last_progress_log_at >= 20.0:
            should_log = True

        if should_log:
            current = payload.get("current")
            total = payload.get("total")
            suffix = ""
            if current is not None and total is not None and float(total or 0) > 0:
                suffix = f" [{int(current)}/{int(total)}]"
            self._append_log(f"[progress] {message}{suffix}")
            self._last_progress_stage = stage
            self._last_progress_message = message
            self._last_progress_log_at = now

        live_preview = payload.get("live_preview")
        if isinstance(live_preview, dict):
            self._update_live_preview(live_preview)

        if fraction <= 0.01 or self.analysis_started_at <= 0:
            self.progress_eta_label.setText("ETA: --")
            return

        elapsed = max(time.perf_counter() - self.analysis_started_at, 0.0)
        remaining = max(elapsed * (1.0 - fraction) / max(fraction, 1e-6), 0.0)
        if remaining >= 60.0:
            eta_text = f"ETA: {remaining / 60.0:.1f} мин"
        else:
            eta_text = f"ETA: {remaining:.0f} сек"
        self.progress_eta_label.setText(eta_text)

    def _on_analysis_finished(self, payload: dict[str, Any]) -> None:
        """Загружает артефакты после завершения анализа."""

        self.current_payload = payload
        self.current_summary = payload.get("summary", {}) or {}
        self._append_log("Анализ завершён. Загружаю таблицы и графики.")
        try:
            self._load_artifacts()
            self.progress_bar.setValue(1000)
            self.progress_stage_label.setText("Анализ завершён.")
            self.progress_eta_label.setText("ETA: 0 сек")
            self.statusBar().showMessage("Анализ завершён успешно.", 10000)
            self.tabs.setCurrentWidget(self.summary_tab)
        except Exception as error:
            self._append_log(traceback.format_exc())
            QMessageBox.critical(self, "Ошибка загрузки результатов", str(error))
            self.statusBar().showMessage("Анализ завершился, но результаты не удалось показать.", 10000)
        finally:
            self.run_button.setEnabled(True)

    def _on_analysis_failed(self, error_text: str) -> None:
        """Показывает ошибку фонового запуска."""

        self.run_button.setEnabled(True)
        self.progress_stage_label.setText("Анализ завершился ошибкой.")
        self._append_log(error_text)
        QMessageBox.critical(self, "Ошибка анализа", error_text)
        self.statusBar().showMessage("Анализ завершился ошибкой.", 10000)

    def _cleanup_worker(self) -> None:
        """Освобождает поток после завершения."""

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
            self.worker_thread = None

    def _load_artifacts(self) -> None:
        """Загружает csv/json/png после выполнения сценария."""

        generated = self.current_summary.get("generated_files", {})
        self.features_df = pd.read_csv(Path(generated["features_csv"]), low_memory=False)
        self.results_df = pd.read_csv(Path(generated["results_csv"]), low_memory=False)
        self.comparison_df = pd.read_csv(Path(generated["comparison_csv"]), low_memory=False)

        self.features_df["_segment_key"] = self.features_df.apply(build_segment_key, axis=1)
        self.results_df["_segment_key"] = self.results_df.apply(build_segment_key, axis=1)
        self.segment_index_map = {
            str(key): int(index)
            for index, key in zip(self.features_df.index.tolist(), self.features_df["_segment_key"].tolist(), strict=False)
        }

        self._populate_summary_tab()
        self._populate_segment_selector()
        self._populate_dynamic_tab()

    def _populate_summary_tab(self) -> None:
        """Заполняет сводную вкладку."""

        parts = [
            f"<h3>Запуск: {self.current_payload.get('dataset', '')} / {self.current_payload.get('mode', '')}</h3>",
            f"<p><b>Динамика:</b> {'включена' if self.current_payload.get('dynamic') else 'выключена'}</p>",
        ]

        ranking = self.current_summary.get("version_ranking", [])
        if ranking:
            parts.append("<b>Ранжирование версий:</b><br>")
            for row in ranking:
                parts.append(f"{row['utility_rank']}. {row['version']} — utility={format_float(row.get('utility_score'), 4)}<br>")

        reliable = self.current_summary.get("reliable_comparisons", [])
        if reliable:
            parts.append("<br><b>Надёжные версии:</b> " + ", ".join(reliable))

        limitations = self.current_summary.get("limitations", [])
        if limitations:
            parts.append("<br><b>Ограничения:</b><br>" + "<br>".join(limitations))

        self.summary_browser.setHtml("".join(parts))
        self.comparison_model.set_frame(self._prepare_comparison_frame())
        self._populate_static_plot_selector()

    def _prepare_comparison_frame(self) -> pd.DataFrame:
        """Формирует компактную таблицу сравнения версий."""

        if self.comparison_df.empty:
            return pd.DataFrame()

        columns = [
            "version",
            "utility_rank",
            "reliability_level",
            "coverage",
            "effect_size",
            "distribution_overlap",
            "relative_sensitivity",
            "stress_baseline_diff",
            "arousal_correlation",
            "valence_correlation",
            "oversmoothing_flag",
        ]
        frame = self.comparison_df[[column for column in columns if column in self.comparison_df.columns]].copy()
        frame = frame.sort_values(["utility_rank", "version"])
        return frame.rename(
            columns={
                "version": "Версия",
                "utility_rank": "Ранг",
                "reliability_level": "Надёжность",
                "coverage": "Покрытие",
                "effect_size": "Effect size",
                "distribution_overlap": "Overlap",
                "relative_sensitivity": "Отн. чувствительность",
                "stress_baseline_diff": "Stress-Baseline",
                "arousal_correlation": "corr(IIS, arousal)",
                "valence_correlation": "corr(Q, valence)",
                "oversmoothing_flag": "Сверхсглаживание",
            }
        )

    def _populate_static_plot_selector(self) -> None:
        """Заполняет список доступных статических графиков."""

        self.static_plot_combo.blockSignals(True)
        self.static_plot_combo.clear()
        dataset = self.current_payload.get("dataset", "")
        mode = self.current_payload.get("mode", "")
        for prefix, title in STATIC_PLOT_SPECS:
            self.static_plot_combo.addItem(title, str(PLOTS_DIR / f"{prefix}_{dataset}_{mode}.png"))
        self.static_plot_combo.blockSignals(False)
        self._update_static_plot()

    def _update_static_plot(self) -> None:
        """Показывает выбранный сводный график."""

        raw_path = self.static_plot_combo.currentData()
        self.static_plot_viewer.set_image(Path(raw_path) if raw_path else None)

    def _populate_segment_selector(self) -> None:
        """Заполняет список сегментов."""

        self.segment_combo.blockSignals(True)
        self.segment_combo.clear()
        for key in self.features_df["_segment_key"].tolist():
            self.segment_combo.addItem(str(key))
        self.segment_combo.blockSignals(False)

        if self.segment_combo.count():
            self.segment_combo.setCurrentIndex(0)
            self._update_segment_views(self.segment_combo.currentText())
        else:
            self.features_model.set_frame(pd.DataFrame())
            self.versions_model.set_frame(pd.DataFrame())
            self.segment_note_browser.setText("Нет сегментов для отображения.")

    def _update_segment_views(self, segment_key: str) -> None:
        """Обновляет таблицы по выбранному сегменту."""

        if not segment_key or segment_key not in self.segment_index_map:
            return

        feature_row = self.features_df.loc[self.segment_index_map[segment_key]]
        version_rows = self.results_df.loc[self.results_df["_segment_key"] == segment_key].copy()
        version_rows["version"] = pd.Categorical(version_rows["version"], categories=list(MODEL_ORDER), ordered=True)
        version_rows = version_rows.sort_values("version")

        self.features_model.set_frame(self._prepare_feature_frame(feature_row))
        self.versions_model.set_frame(self._prepare_versions_frame(version_rows))
        self.segment_note_browser.setHtml(self._build_segment_note(feature_row, version_rows))

    def _prepare_feature_frame(self, row: pd.Series) -> pd.DataFrame:
        """Собирает входные признаки выбранного сегмента в вертикальную таблицу."""

        records: list[dict[str, Any]] = []
        for column in SEGMENT_FEATURE_COLUMNS:
            if column not in row.index:
                continue
            value = row.get(column)
            provenance = row.get(f"prov_{column}", "")
            source = row.get(f"source_{column}", "")
            if pd.isna(value) and not provenance and not source:
                continue
            records.append({"Признак": column, "Значение": value, "Происхождение": provenance, "Источник": source})
        return pd.DataFrame.from_records(records)

    def _prepare_versions_frame(self, version_rows: pd.DataFrame) -> pd.DataFrame:
        """Готовит короткую таблицу по всем версиям модели на одном сегменте."""

        if version_rows.empty:
            return pd.DataFrame()
        frame = version_rows[list(VERSION_RESULT_COLUMNS)].copy()
        return frame.rename(
            columns={
                "version": "Версия",
                "coverage_ratio": "Покрытие",
                "active_component_count": "Активных блоков",
            }
        )

    def _build_segment_note(self, feature_row: pd.Series, version_rows: pd.DataFrame) -> str:
        """Строит текстовое пояснение по выбранному сегменту."""

        parts = [
            "<h3>Пояснение сегмента</h3>",
            f"<b>Ключ:</b> {feature_row['_segment_key']}<br>",
            f"<b>Датасет:</b> {feature_row.get('dataset', '')}<br>",
            f"<b>Метка:</b> {feature_row.get('label', '')}<br>",
            f"<b>Источник записи:</b> {feature_row.get('source_record_id', '')}<br>",
        ]

        for _, row in version_rows.iterrows():
            explanation = safe_json_loads(row.get("score_explanation_json")) or {}
            top_contributors = explanation.get("top_contributors", [])
            parts.extend(
                [
                    "<hr>",
                    f"<b>{row.get('version', '')}</b><br>",
                    f"IIS: {format_float(row.get('IIS'))}<br>",
                    f"A={format_float(row.get('A'))}, Gamma={format_float(row.get('Gamma'))}, H={format_float(row.get('H'))}, V={format_float(row.get('V'))}, Q={format_float(row.get('Q'))}, K={format_float(row.get('K'))}<br>",
                    f"Покрытие: {format_float(row.get('coverage_ratio'))}<br>",
                    f"Активные блоки: {row.get('active_components_json', '')}<br>",
                    f"Отсутствуют: {row.get('missing_components_json', '')}<br>",
                    f"Главные вклады: {', '.join(top_contributors) if top_contributors else 'нет'}<br>",
                    f"Формула: {row.get('formula_note', '')}<br>",
                ]
            )
            if row.get("insufficient_reason"):
                parts.append(f"Ограничение: {row.get('insufficient_reason')}<br>")
        return "".join(parts)

    def _populate_dynamic_tab(self) -> None:
        """Загружает динамические артефакты, если они есть."""

        self.dynamic_df = pd.DataFrame()
        if not self.current_payload.get("dynamic"):
            self.dynamic_note_browser.setText("Текущий запуск был без динамического сценария.")
            self.dynamic_model.set_frame(pd.DataFrame())
            self.dynamic_record_combo.clear()
            self.dynamic_plot_viewer.set_image(None)
            return

        dynamic_outputs = self.current_summary.get("dynamic_outputs", {})
        dynamic_csv = dynamic_outputs.get("dynamic_csv")
        dynamic_plot = dynamic_outputs.get("dynamic_plot")
        if dynamic_csv and Path(dynamic_csv).exists():
            self.dynamic_df = pd.read_csv(dynamic_csv, low_memory=False)
        self.dynamic_plot_viewer.set_image(Path(dynamic_plot) if dynamic_plot else None)

        self.dynamic_note_browser.setHtml(
            "".join(
                [
                    "<h3>Причинная динамика</h3>",
                    f"<b>Версия:</b> {dynamic_outputs.get('focus_version', self.current_payload.get('focus_version', ''))}<br>",
                    f"<b>Окно:</b> {self.current_payload.get('window_seconds', 0.0):.1f} c<br>",
                    f"<b>Шаг:</b> {self.current_payload.get('step_seconds', 0.0):.1f} c<br>",
                    "IIS_dynamic использует только прошлые окна: быстрый вход в ухудшение и медленное восстановление.<br>",
                    f"<b>CSV:</b> {dynamic_csv or 'нет'}<br>",
                    f"<b>PNG:</b> {dynamic_plot or 'нет'}<br>",
                ]
            )
        )

        self.dynamic_record_combo.blockSignals(True)
        self.dynamic_record_combo.clear()
        if not self.dynamic_df.empty and "_source_key" in self.dynamic_df.columns:
            for key in self.dynamic_df["_source_key"].dropna().astype(str).unique().tolist():
                self.dynamic_record_combo.addItem(key)
        self.dynamic_record_combo.blockSignals(False)

        if self.dynamic_record_combo.count():
            self.dynamic_record_combo.setCurrentIndex(0)
            self._update_dynamic_table(self.dynamic_record_combo.currentText())
        else:
            self.dynamic_model.set_frame(pd.DataFrame())

        self._populate_intervention_selector()

    def _update_dynamic_table(self, record_key: str) -> None:
        """Показывает строки динамики по одной записи."""

        if self.dynamic_df.empty or not record_key:
            self.dynamic_model.set_frame(pd.DataFrame())
            return

        frame = self.dynamic_df.loc[self.dynamic_df["_source_key"].astype(str) == str(record_key)].copy()
        if frame.empty:
            self.dynamic_model.set_frame(pd.DataFrame())
            return

        preview = frame[[column for column in DYNAMIC_PREVIEW_COLUMNS if column in frame.columns]].head(400).copy()
        self.dynamic_model.set_frame(preview)

    def _populate_intervention_selector(self) -> None:
        """Заполняет список записей для интервенций."""

        self.intervention_record_combo.blockSignals(True)
        self.intervention_record_combo.clear()
        source_ids = []
        if not self.features_df.empty and "source_record_id" in self.features_df.columns:
            source_ids = self.features_df["source_record_id"].dropna().astype(str).unique().tolist()
        for key in source_ids:
            self.intervention_record_combo.addItem(key)
        self.intervention_record_combo.blockSignals(False)

        if self.intervention_record_combo.count():
            self.intervention_record_combo.setCurrentIndex(0)
            self._sync_intervention_time_range(self.intervention_record_combo.currentText())
        else:
            self.intervention_model.set_frame(pd.DataFrame())
            self.intervention_plot_viewer.set_image(None)

    def _sync_intervention_time_range(self, source_key: str) -> None:
        """Подставляет рекомендуемый временной интервал по выбранной записи."""

        if not source_key or self.features_df.empty:
            return
        record_df = self.features_df.loc[self.features_df["source_record_id"].astype(str) == str(source_key)].copy()
        if record_df.empty:
            return
        starts = pd.to_numeric(record_df["window_start_sec"], errors="coerce").dropna()
        ends = pd.to_numeric(record_df["window_end_sec"], errors="coerce").dropna()
        if starts.empty or ends.empty:
            return
        self.intervention_start_spin.setValue(float(starts.min()))
        self.intervention_end_spin.setValue(float(min(starts.min() + max((ends.max() - starts.min()) * 0.25, 4.0), ends.max())))

    def _run_intervention(self) -> None:
        """Запускает псевдо-причинную интервенцию по выбранной записи."""

        if self.features_df.empty or not self.current_payload:
            QMessageBox.warning(self, "Нет данных", "Сначала выполните обычный анализ с динамикой.")
            return

        simulator = IISInterventionSimulator(output_dir=OUTPUT_DIR, plots_dir=PLOTS_DIR)
        payload = simulator.simulate(
            features_df=self.features_df,
            dataset=self.current_payload.get("dataset", ""),
            mode=self.current_payload.get("mode", ""),
            focus_version=self.focus_version_combo.currentText(),
            source_key=self.intervention_record_combo.currentText(),
            target_column=self.intervention_target_combo.currentText(),
            start_time_sec=float(self.intervention_start_spin.value()),
            end_time_sec=float(self.intervention_end_spin.value()),
            magnitude=float(self.intervention_magnitude_spin.value()),
            operation=self.intervention_operation_combo.currentText(),
        )
        if not payload:
            QMessageBox.warning(self, "Интервенция не выполнена", "Не удалось построить сценарий для выбранных параметров.")
            return

        self.intervention_df = payload["intervention_dataframe"]
        preview_columns = [
            column
            for column in (
                "time_sec",
                "label",
                self.intervention_target_combo.currentText(),
                f"{self.intervention_target_combo.currentText()}_intervened",
                "IIS",
                "IIS_intervened",
                "IIS_dynamic",
                "IIS_dynamic_intervened",
                "intervention_mask",
            )
            if column in self.intervention_df.columns
        ]
        self.intervention_model.set_frame(self.intervention_df[preview_columns].head(500).copy())
        self.intervention_plot_viewer.set_image(Path(payload["intervention_plot"]))
        self.intervention_note_browser.setHtml(
            "".join(
                [
                    "<h3>Псевдо-причинная интервенция</h3>",
                    f"<b>Запись:</b> {self.intervention_record_combo.currentText()}<br>",
                    f"<b>Признак:</b> {self.intervention_target_combo.currentText()}<br>",
                    f"<b>Операция:</b> {self.intervention_operation_combo.currentText()}<br>",
                    f"<b>Амплитуда:</b> {self.intervention_magnitude_spin.value():.4f}<br>",
                    f"<b>Интервал:</b> {self.intervention_start_spin.value():.1f} - {self.intervention_end_spin.value():.1f} c<br>",
                    "Интервенция меняет извлечённый признак окна и затем пересчитывает IIS и IIS_dynamic. "
                    "Это модельный стресс-тест, а не заявление о прямой физиологической причинности.<br>",
                    f"<b>CSV:</b> {payload.get('intervention_csv', '')}<br>",
                    f"<b>PNG:</b> {payload.get('intervention_plot', '')}<br>",
                ]
            )
        )


def main() -> int:
    """Точка входа GUI."""

    ensure_runtime_directories()
    app = QApplication(sys.argv)
    app.setApplicationName("ИИС Аналитика")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
