
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess


class ARIMAGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("ARMA / ARIMA Analyzer")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 750)

        # Generated data
        self.arma_values = None
        self.series = None
        self.diff_series = None

        # Trend analysis
        self.trend_values = None
        self.detrended_series = None
        self.trend_slope = None
        self.trend_intercept = None

        # ACF and PACF (original and detrended series)
        self.acf_values = None
        self.pacf_values = None
        self.detrended_acf_values = None
        self.detrended_pacf_values = None
        self.acf_table = None

        self.figures = {}

        self.setup_style()
        self.create_interface()

    # =====================================================
    # GUI Style
    # =====================================================

    def setup_style(self):

        # Visual styling only: the analysis and calculation code is unchanged.
        style = ttk.Style()
        style.theme_use("clam")

        self.root.configure(bg="#111827")

        # Color palette
        background = "#111827"
        surface = "#1F2937"
        surface_alt = "#273449"
        foreground = "#E5E7EB"
        muted = "#9CA3AF"
        accent = "#38BDF8"
        accent_hover = "#0EA5E9"
        border = "#374151"
        input_bg = "#0F172A"
        selected = "#164E63"

        # General widgets
        style.configure(
            ".",
            background=background,
            foreground=foreground,
            font=("Segoe UI", 10)
        )

        style.configure("TFrame", background=background)
        style.configure(
            "TLabel",
            background=background,
            foreground=foreground,
            font=("Segoe UI", 10)
        )

        style.configure(
            "TLabelframe",
            background=background,
            bordercolor=border,
            relief="solid",
            borderwidth=1
        )
        style.configure(
            "TLabelframe.Label",
            background=background,
            foreground=accent,
            font=("Segoe UI", 10, "bold")
        )

        # Notebook and tabs
        style.configure(
            "TNotebook",
            background=background,
            borderwidth=0,
            tabmargins=(4, 5, 4, 0)
        )
        style.configure(
            "TNotebook.Tab",
            background=surface,
            foreground=muted,
            padding=(18, 10),
            font=("Segoe UI", 10, "bold"),
            borderwidth=0
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", accent), ("active", surface_alt)],
            foreground=[("selected", "#082F49"), ("active", foreground)]
        )

        # Input fields
        style.configure(
            "TEntry",
            fieldbackground=input_bg,
            foreground=foreground,
            insertcolor=foreground,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            padding=(9, 7)
        )
        style.map(
            "TEntry",
            bordercolor=[("focus", accent)],
            lightcolor=[("focus", accent)],
            darkcolor=[("focus", accent)]
        )

        # Buttons
        style.configure(
            "TButton",
            background=surface_alt,
            foreground=foreground,
            padding=(14, 9),
            borderwidth=0,
            focusthickness=0,
            font=("Segoe UI", 10, "bold")
        )
        style.map(
            "TButton",
            background=[
                ("disabled", surface),
                ("pressed", accent_hover),
                ("active", accent)
            ],
            foreground=[
                ("disabled", muted),
                ("pressed", "#082F49"),
                ("active", "#082F49")
            ]
        )

        # Results table
        style.configure(
            "Treeview",
            background=input_bg,
            fieldbackground=input_bg,
            foreground=foreground,
            rowheight=30,
            bordercolor=border,
            relief="flat",
            font=("Segoe UI", 10)
        )
        style.configure(
            "Treeview.Heading",
            background=surface_alt,
            foreground=accent,
            padding=(10, 9),
            relief="flat",
            font=("Segoe UI", 10, "bold")
        )
        style.map(
            "Treeview",
            background=[("selected", selected)],
            foreground=[("selected", "#FFFFFF")]
        )
        style.map(
            "Treeview.Heading",
            background=[("active", surface)]
        )

        # Scrollbars
        style.configure(
            "Vertical.TScrollbar",
            background=surface_alt,
            troughcolor=background,
            bordercolor=background,
            arrowcolor=foreground
        )
        style.configure(
            "Horizontal.TScrollbar",
            background=surface_alt,
            troughcolor=background,
            bordercolor=background,
            arrowcolor=foreground
        )

    # =====================================================
    # Main Interface
    # =====================================================

    def create_interface(self):

        header = ttk.Frame(self.root, padding=(22, 16, 22, 10))
        header.pack(fill=tk.X)

        title = ttk.Label(
            header,
            text="ARMA / ARIMA  |  Time Series Analysis",
            font=("Segoe UI", 21, "bold"),
            foreground="#38BDF8"
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            header,
            text="Generate a process, inspect its autocorrelation, and compare results before and after detrending.",
            foreground="#9CA3AF",
            font=("Segoe UI", 10)
        )
        subtitle.pack(anchor="w", pady=(4, 0))

        self.notebook = ttk.Notebook(self.root)

        self.notebook.pack(
            fill=tk.BOTH,
            expand=True,
            padx=12,
            pady=8
        )

        self.input_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)

        self.notebook.add(
            self.input_tab,
            text="Input Parameters"
        )

        self.notebook.add(
            self.results_tab,
            text="Results"
        )

        self.create_input_tab()
        self.create_results_tab()

    # =====================================================
    # Input Parameters
    # =====================================================

    def create_input_tab(self):

        main_frame = ttk.Frame(
            self.input_tab,
            padding=20
        )

        main_frame.pack(
            fill=tk.BOTH,
            expand=True
        )

        # ---------------------------------------------
        # ARIMA Parameters
        # ---------------------------------------------

        arima_frame = ttk.LabelFrame(
            main_frame,
            text="ARIMA Parameters",
            padding=15
        )

        arima_frame.pack(
            fill=tk.X,
            pady=8
        )

        self.entries = {}

        arima_parameters = [

            ("p - AR Order", "1"),
            ("d_model - Integration Order", "1"),

            ("q - MA Order", "0"),
            ("d_diff - Differencing Operations", "3"),

            ("Number of Samples (n)", "500"),
            ("Burn-in", "0"),

            ("Random Seed", "42"),
            ("Maximum Lags", "16")

        ]

        for i, (label, default) in enumerate(arima_parameters):

            row = i // 2
            col = (i % 2) * 2

            ttk.Label(
                arima_frame,
                text=label
            ).grid(
                row=row,
                column=col,
                sticky="w",
                padx=8,
                pady=8
            )

            entry = ttk.Entry(
                arima_frame,
                width=18
            )

            entry.insert(0, default)

            entry.grid(
                row=row,
                column=col + 1,
                padx=8,
                pady=8
            )

            self.entries[label] = entry

        # ---------------------------------------------
        # AR and MA Coefficients
        # ---------------------------------------------

        coefficient_frame = ttk.LabelFrame(
            main_frame,
            text="AR and MA Coefficients",
            padding=15
        )

        coefficient_frame.pack(
            fill=tk.X,
            pady=8
        )

        ttk.Label(
            coefficient_frame,
            text="AR Coefficients (comma separated)"
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=8,
            pady=8
        )

        self.ar_entry = ttk.Entry(
            coefficient_frame,
            width=35
        )

        self.ar_entry.insert(0, "0.8")

        self.ar_entry.grid(
            row=0,
            column=1,
            padx=8,
            pady=8
        )

        ttk.Label(
            coefficient_frame,
            text="MA Coefficients (comma separated)"
        ).grid(
            row=1,
            column=0,
            sticky="w",
            padx=8,
            pady=8
        )

        self.ma_entry = ttk.Entry(
            coefficient_frame,
            width=35
        )

        self.ma_entry.insert(0, "")

        self.ma_entry.grid(
            row=1,
            column=1,
            padx=8,
            pady=8
        )

        # ---------------------------------------------
        # Noise Parameters
        # ---------------------------------------------

        noise_frame = ttk.LabelFrame(
            main_frame,
            text="Noise and Drift Parameters",
            padding=15
        )

        noise_frame.pack(
            fill=tk.X,
            pady=8
        )

        noise_parameters = [

            ("Noise Mean", "0.0"),
            ("Noise Variance", "1.0"),
            ("Drift", "0.15")

        ]

        for i, (label, default) in enumerate(noise_parameters):

            ttk.Label(
                noise_frame,
                text=label
            ).grid(
                row=0,
                column=i * 2,
                padx=8,
                pady=8,
                sticky="w"
            )

            entry = ttk.Entry(
                noise_frame,
                width=15
            )

            entry.insert(0, default)

            entry.grid(
                row=0,
                column=i * 2 + 1,
                padx=8,
                pady=8
            )

            self.entries[label] = entry

        # ---------------------------------------------
        # Generate Button
        # ---------------------------------------------

        button_frame = ttk.Frame(main_frame)

        button_frame.pack(pady=20)

        generate_button = ttk.Button(
            button_frame,
            text="Generate and Analyze",
            command=self.generate_results
        )

        generate_button.pack(
            ipadx=25,
            ipady=7
        )

    # =====================================================
    # Results Interface
    # =====================================================

    def create_results_tab(self):

        self.result_notebook = ttk.Notebook(
            self.results_tab
        )

        self.result_notebook.pack(
            fill=tk.BOTH,
            expand=True,
            padx=8,
            pady=8
        )

        # ---------------------------------------------
        # ARMA Tab
        # ---------------------------------------------

        self.arma_tab = ttk.Frame(
            self.result_notebook
        )

        self.result_notebook.add(
            self.arma_tab,
            text="ARMA Plot"
        )

        self.arma_plot_frame = ttk.Frame(
            self.arma_tab
        )

        self.arma_plot_frame.pack(
            fill=tk.BOTH,
            expand=True
        )

        # ---------------------------------------------
        # ACF Tab
        # ---------------------------------------------

        self.acf_tab = ttk.Frame(
            self.result_notebook
        )

        self.result_notebook.add(
            self.acf_tab,
            text="ACF Plot"
        )

        self.acf_plot_frame = ttk.Frame(
            self.acf_tab
        )

        self.acf_plot_frame.pack(
            fill=tk.BOTH,
            expand=True
        )

        # ---------------------------------------------
        # PACF Tab
        # ---------------------------------------------

        self.pacf_tab = ttk.Frame(
            self.result_notebook
        )

        self.result_notebook.add(
            self.pacf_tab,
            text="PACF Plot"
        )

        self.pacf_plot_frame = ttk.Frame(
            self.pacf_tab
        )

        self.pacf_plot_frame.pack(
            fill=tk.BOTH,
            expand=True
        )

        # ---------------------------------------------
        # ARIMA Series Tab
        # ---------------------------------------------

        self.series_tab = ttk.Frame(
            self.result_notebook
        )

        self.result_notebook.add(
            self.series_tab,
            text="ARIMA / Differenced Series"
        )

        self.series_plot_frame = ttk.Frame(
            self.series_tab
        )

        self.series_plot_frame.pack(
            fill=tk.BOTH,
            expand=True
        )

        # ---------------------------------------------
        # Trend Analysis Tab
        # ---------------------------------------------

        self.trend_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.trend_tab, text="Trend Analysis")

        self.trend_plot_frame = ttk.Frame(self.trend_tab)
        self.trend_plot_frame.pack(fill=tk.BOTH, expand=True)

        # ---------------------------------------------
        # Detrended ACF / PACF Tabs
        # ---------------------------------------------

        self.detrended_acf_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.detrended_acf_tab, text="Detrended ACF")
        self.detrended_acf_plot_frame = ttk.Frame(self.detrended_acf_tab)
        self.detrended_acf_plot_frame.pack(fill=tk.BOTH, expand=True)

        self.detrended_pacf_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.add(self.detrended_pacf_tab, text="Detrended PACF")
        self.detrended_pacf_plot_frame = ttk.Frame(self.detrended_pacf_tab)
        self.detrended_pacf_plot_frame.pack(fill=tk.BOTH, expand=True)

        # ---------------------------------------------
        # ACF / PACF Table Tab
        # ---------------------------------------------

        self.table_tab = ttk.Frame(
            self.result_notebook
        )

        self.result_notebook.add(
            self.table_tab,
            text="ACF / PACF Table"
        )

        self.create_table()

        # ---------------------------------------------
        # Export Buttons
        # ---------------------------------------------

        save_frame = ttk.Frame(
            self.results_tab
        )

        save_frame.pack(
            fill=tk.X,
            pady=5
        )

        ttk.Button(
            save_frame,
            text="Save ARMA",
            command=lambda: self.save_figure("arma")
        ).pack(
            side=tk.LEFT,
            padx=5
        )

        ttk.Button(
            save_frame,
            text="Save ACF",
            command=lambda: self.save_figure("acf")
        ).pack(
            side=tk.LEFT,
            padx=5
        )

        ttk.Button(
            save_frame,
            text="Save PACF",
            command=lambda: self.save_figure("pacf")
        ).pack(
            side=tk.LEFT,
            padx=5
        )

        ttk.Button(
            save_frame,
            text="Save ARIMA Series",
            command=lambda: self.save_figure("series")
        ).pack(
            side=tk.LEFT,
            padx=5
        )

        ttk.Button(
            save_frame,
            text="Save Trend Analysis",
            command=lambda: self.save_figure("trend")
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            save_frame,
            text="Save Detrended ACF",
            command=lambda: self.save_figure("detrended_acf")
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            save_frame,
            text="Save Detrended PACF",
            command=lambda: self.save_figure("detrended_pacf")
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            save_frame,
            text="Export Table",
            command=self.export_table
        ).pack(
            side=tk.LEFT,
            padx=5
        )

    # =====================================================
    # ACF / PACF Table
    # =====================================================

    def create_table(self):

        columns = (
            "Lag (k)",
            "Original ACF",
            "Original PACF",
            "Detrended ACF",
            "Detrended PACF"
        )

        self.tree = ttk.Treeview(
            self.table_tab,
            columns=columns,
            show="headings"
        )

        for column in columns:

            self.tree.heading(
                column,
                text=column
            )

            self.tree.column(
                column,
                width=190,
                anchor="center"
            )

        self.tree.pack(
            fill=tk.BOTH,
            expand=True,
            padx=15,
            pady=15
        )

    # =====================================================
    # Parse Coefficients
    # =====================================================

    def parse_coefficients(self, text):

        text = text.strip()

        if not text:
            return []

        try:

            return [
                float(value.strip())
                for value in text.split(",")
            ]

        except ValueError:

            raise ValueError(
                "Coefficients must be numbers separated by commas."
            )

    # =====================================================
    # Read Parameters
    # =====================================================

    def read_parameters(self):

        values = {}

        for label, entry in self.entries.items():

            values[label] = float(
                entry.get().strip()
            )

        integer_parameters = [

            "p - AR Order",
            "d_model - Integration Order",
            "q - MA Order",
            "d_diff - Differencing Operations",
            "Number of Samples (n)",
            "Burn-in",
            "Random Seed",
            "Maximum Lags"

        ]

        for key in integer_parameters:

            if not values[key].is_integer():

                raise ValueError(
                    f"{key} must be an integer."
                )

            values[key] = int(values[key])

        return values

    # =====================================================
    # Generate Results
    # =====================================================

    def generate_results(self):

        try:

            params = self.read_parameters()

            # -----------------------------------------
            # Parameters
            # -----------------------------------------

            p = params["p - AR Order"]

            d_model = params[
                "d_model - Integration Order"
            ]

            q = params["q - MA Order"]

            d_diff = params[
                "d_diff - Differencing Operations"
            ]

            n = params["Number of Samples (n)"]

            burn_in = params["Burn-in"]

            seed = params["Random Seed"]

            max_lags = params["Maximum Lags"]

            # -----------------------------------------
            # Coefficients
            # -----------------------------------------

            ar_coefficients = self.parse_coefficients(
                self.ar_entry.get()
            )

            ma_coefficients = self.parse_coefficients(
                self.ma_entry.get()
            )

            # -----------------------------------------
            # Noise Parameters
            # -----------------------------------------

            noise_mean = params["Noise Mean"]

            noise_variance = params["Noise Variance"]

            drift = params["Drift"]

            # =========================================
            # Validation
            # =========================================

            if p != len(ar_coefficients):

                raise ValueError(
                    "AR coefficient count must equal p."
                )

            if q != len(ma_coefficients):

                raise ValueError(
                    "MA coefficient count must equal q."
                )

            if min(p, d_model, q, d_diff) < 0:

                raise ValueError(
                    "Model orders must be non-negative."
                )

            if n <= 0:

                raise ValueError(
                    "Number of samples must be positive."
                )

            if burn_in < 0:

                raise ValueError(
                    "Burn-in cannot be negative."
                )

            if noise_variance < 0:

                raise ValueError(
                    "Noise variance cannot be negative."
                )

            if max_lags < 1:

                raise ValueError(
                    "Maximum lags must be at least 1."
                )

            # =========================================
            # Generate ARMA Process
            # =========================================

            rng = np.random.default_rng(seed)

            ar = np.r_[
                1.0,
                -np.array(ar_coefficients)
            ]

            ma = np.r_[
                1.0,
                np.array(ma_coefficients)
            ]

            arma_process = ArmaProcess(
                ar,
                ma
            )

            if not arma_process.isstationary:

                raise ValueError(
                    "AR coefficients must define a stationary process."
                )

            def generate_noise(size):

                return rng.normal(
                    noise_mean,
                    np.sqrt(noise_variance),
                    size
                )

            arma_values = arma_process.generate_sample(
                nsample=n + burn_in,
                distrvs=generate_noise,
                burnin=burn_in
            )

            arma_values += drift

            # =========================================
            # Generate ARIMA Series
            # =========================================

            series = arma_values.copy()

            for _ in range(d_model):

                series = np.cumsum(series)

            series = series[-n:]

            # =========================================
            # Independent Differencing
            # =========================================

            diff_series = series.copy()

            for _ in range(d_diff):

                diff_series = np.diff(diff_series)

            # =========================================
            # Linear Trend Estimation and Detrending
            # =========================================

            time_index = np.arange(len(series), dtype=float)
            trend_slope, trend_intercept = np.polyfit(time_index, series, 1)
            trend_values = trend_slope * time_index + trend_intercept
            detrended_series = series - trend_values

            # =========================================
            # ACF Calculation (Original Series)
            # =========================================

            acf_values = acf(
                series,
                nlags=max_lags,
                fft=False
            )

            # =========================================
            # PACF Calculation
            # =========================================

            if max_lags >= len(series) / 2:

                raise ValueError(
                    "Maximum lags must be less than half the series length."
                )

            pacf_values = pacf(
                series,
                nlags=max_lags,
                method="ywm"
            )

            # =========================================
            # ACF / PACF (Detrended Series)
            # =========================================

            detrended_acf_values = acf(
                detrended_series,
                nlags=max_lags,
                fft=False
            )

            detrended_pacf_values = pacf(
                detrended_series,
                nlags=max_lags,
                method="ywm"
            )

            # =========================================
            # ACF / PACF Table
            # =========================================

            acf_table = pd.DataFrame({
                "Lag (k)": range(max_lags + 1),
                "Original ACF": acf_values,
                "Original PACF": pacf_values,
                "Detrended ACF": detrended_acf_values,
                "Detrended PACF": detrended_pacf_values
            })

            # =========================================
            # Store Results
            # =========================================

            self.arma_values = arma_values

            self.series = series

            self.diff_series = diff_series

            self.trend_values = trend_values
            self.detrended_series = detrended_series
            self.trend_slope = trend_slope
            self.trend_intercept = trend_intercept

            self.acf_values = acf_values
            self.pacf_values = pacf_values
            self.detrended_acf_values = detrended_acf_values
            self.detrended_pacf_values = detrended_pacf_values

            self.acf_table = acf_table

            self.current_parameters = {

                "p": p,
                "d_model": d_model,
                "q": q,
                "d_diff": d_diff,
                "max_lags": max_lags

            }

            # =========================================
            # Display Results
            # =========================================

            self.update_table()

            self.update_plots()

            self.notebook.select(
                self.results_tab
            )

            messagebox.showinfo(
                "Success",
                "Analysis completed: original and detrended series, ACF and PACF are ready."
            )

        except Exception as error:

            messagebox.showerror(
                "Error",
                str(error)
            )

    # =====================================================
    # Update Table
    # =====================================================

    def update_table(self):

        for item in self.tree.get_children():

            self.tree.delete(item)

        for _, row in self.acf_table.iterrows():

            self.tree.insert(
                "",
                tk.END,
                values=(
                    int(row["Lag (k)"]),
                    f'{row["Original ACF"]:.6f}',
                    f'{row["Original PACF"]:.6f}',
                    f'{row["Detrended ACF"]:.6f}',
                    f'{row["Detrended PACF"]:.6f}'
                )
            )

    # =====================================================
    # ARMA Plot
    # =====================================================

    def create_arma_plot(self):

        p = self.current_parameters["p"]

        q = self.current_parameters["q"]

        # Use the last n samples to match the ARIMA series length
        arma_series = self.arma_values[-len(self.series):]

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            arma_series,
            linewidth=1,
            label="ARMA Process"
        )

        ax.set_title(
            f"Generated ARMA({p}, {q}) Process"
        )

        ax.set_xlabel("Time")

        ax.set_ylabel("Value")

        ax.grid(
            True,
            alpha=0.3
        )

        ax.legend()

        fig.tight_layout()

        return fig

    # =====================================================
    # ACF Plot
    # =====================================================

    def create_acf_plot(self):

        max_lags = self.current_parameters["max_lags"]

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        plot_acf(
            self.series,
            lags=max_lags,
            ax=ax,
            alpha=0.05
        )

        ax.set_title(
            "Autocorrelation Function (ACF)"
        )

        ax.set_xlabel("Lag (k)")

        ax.set_ylabel("Autocorrelation")

        ax.grid(
            True,
            alpha=0.3
        )

        fig.tight_layout()

        return fig

    # =====================================================
    # PACF Plot
    # =====================================================

    def create_pacf_plot(self):

        max_lags = self.current_parameters["max_lags"]

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        plot_pacf(
            self.series,
            lags=max_lags,
            ax=ax,
            alpha=0.05,
            method="ywm"
        )

        ax.set_title(
            "Partial Autocorrelation Function (PACF)"
        )

        ax.set_xlabel("Lag (k)")

        ax.set_ylabel("Partial Autocorrelation")

        ax.grid(
            True,
            alpha=0.3
        )

        fig.tight_layout()

        return fig

    # =====================================================
    # ARIMA and Differenced Series Plot
    # =====================================================

    def create_series_plot(self):

        p = self.current_parameters["p"]

        d_model = self.current_parameters["d_model"]

        q = self.current_parameters["q"]

        d_diff = self.current_parameters["d_diff"]

        fig = Figure(
            figsize=(11, 8),
            dpi=100
        )

        # ---------------------------------------------
        # ARIMA Series
        # ---------------------------------------------

        ax1 = fig.add_subplot(211)

        ax1.plot(
            self.series,
            linewidth=1,
            label="Generated ARIMA Series"
        )

        ax1.set_title(
            f"Generated ARIMA({p}, {d_model}, {q}) Series"
        )

        ax1.set_xlabel("Time")

        ax1.set_ylabel("Value")

        ax1.grid(
            True,
            alpha=0.3
        )

        ax1.legend()

        # ---------------------------------------------
        # Differenced Series
        # ---------------------------------------------

        ax2 = fig.add_subplot(212)

        ax2.plot(
            self.diff_series,
            linewidth=1,
            label="Differenced Series"
        )

        ax2.set_title(
            f"Series After {d_diff} Differencing Operations"
        )

        ax2.set_xlabel("Time")

        ax2.set_ylabel("Difference")

        ax2.grid(
            True,
            alpha=0.3
        )

        ax2.legend()

        fig.tight_layout()

        return fig

    # =====================================================
    # Trend and Detrended Diagnostics
    # =====================================================

    def create_trend_plot(self):

        fig = Figure(figsize=(11, 8), dpi=100)
        ax1 = fig.add_subplot(211)
        ax1.plot(self.series, linewidth=1, label="Original Series", alpha=0.8)
        ax1.plot(self.trend_values, linewidth=2, label="Estimated Linear Trend")
        ax1.set_title(
            f"Linear Trend: y = {self.trend_slope:.6f}t + {self.trend_intercept:.6f}"
        )
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Value")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = fig.add_subplot(212)
        ax2.plot(self.detrended_series, linewidth=1, label="Detrended Series")
        ax2.axhline(0, linewidth=1, linestyle="--", alpha=0.7)
        ax2.set_title("Series After Removing the Estimated Linear Trend")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Detrended Value")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        mean_value = np.mean(self.series)
        std_value = np.std(self.series, ddof=1) if len(self.series) > 1 else 0.0
        detrended_mean = np.mean(self.detrended_series)
        detrended_std = np.std(self.detrended_series, ddof=1) if len(self.detrended_series) > 1 else 0.0
        fig.suptitle(
            f"Original mean={mean_value:.4f} | Original std={std_value:.4f}     "
            f"Detrended mean={detrended_mean:.4f} | Detrended std={detrended_std:.4f}",
            fontsize=10
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig

    def create_detrended_acf_plot(self):

        max_lags = self.current_parameters["max_lags"]
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_acf(self.detrended_series, lags=max_lags, ax=ax, alpha=0.05)
        ax.set_title("ACF of Detrended Series")
        ax.set_xlabel("Lag (k)")
        ax.set_ylabel("Autocorrelation")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def create_detrended_pacf_plot(self):

        max_lags = self.current_parameters["max_lags"]
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_pacf(
            self.detrended_series,
            lags=max_lags,
            ax=ax,
            alpha=0.05,
            method="ywm"
        )
        ax.set_title("PACF of Detrended Series")
        ax.set_xlabel("Lag (k)")
        ax.set_ylabel("Partial Autocorrelation")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    # =====================================================
    # Display Figure in GUI
    # =====================================================

    def show_figure(self, figure, frame):

        for widget in frame.winfo_children():

            widget.destroy()

        canvas = FigureCanvasTkAgg(
            figure,
            master=frame
        )

        canvas.draw()

        canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        return canvas

    # =====================================================
    # Update All Plots
    # =====================================================

    def update_plots(self):

        # ARMA Plot
        self.figures["arma"] = self.create_arma_plot()

        self.show_figure(
            self.figures["arma"],
            self.arma_plot_frame
        )

        # ACF Plot
        self.figures["acf"] = self.create_acf_plot()

        self.show_figure(
            self.figures["acf"],
            self.acf_plot_frame
        )

        # PACF Plot
        self.figures["pacf"] = self.create_pacf_plot()

        self.show_figure(
            self.figures["pacf"],
            self.pacf_plot_frame
        )

        # ARIMA and Differenced Series
        self.figures["series"] = self.create_series_plot()

        self.show_figure(
            self.figures["series"],
            self.series_plot_frame
        )

        # Trend and detrended diagnostics
        self.figures["trend"] = self.create_trend_plot()
        self.show_figure(self.figures["trend"], self.trend_plot_frame)

        self.figures["detrended_acf"] = self.create_detrended_acf_plot()
        self.show_figure(
            self.figures["detrended_acf"],
            self.detrended_acf_plot_frame
        )

        self.figures["detrended_pacf"] = self.create_detrended_pacf_plot()
        self.show_figure(
            self.figures["detrended_pacf"],
            self.detrended_pacf_plot_frame
        )

    # =====================================================
    # Save Figures
    # =====================================================

    def save_figure(self, figure_name):

        if figure_name not in self.figures:

            messagebox.showwarning(
                "Warning",
                "Generate results first."
            )

            return

        file_path = filedialog.asksaveasfilename(

            defaultextension=".png",

            filetypes=[

                ("PNG Image", "*.png"),
                ("PDF File", "*.pdf"),
                ("SVG File", "*.svg")

            ],

            initialfile=f"{figure_name}_plot.png"

        )

        if not file_path:

            return

        self.figures[figure_name].savefig(
            file_path,
            dpi=300,
            bbox_inches="tight"
        )

        messagebox.showinfo(
            "Saved",
            f"Plot saved successfully:\n{file_path}"
        )

    # =====================================================
    # Export ACF / PACF Table
    # =====================================================

    def export_table(self):

        if self.acf_table is None:

            messagebox.showwarning(
                "Warning",
                "Generate results first."
            )

            return

        file_path = filedialog.asksaveasfilename(

            defaultextension=".csv",

            filetypes=[

                ("CSV File", "*.csv"),
                ("Excel File", "*.xlsx")

            ],

            initialfile="original_detrended_acf_pacf.csv"

        )

        if not file_path:

            return

        try:

            if file_path.endswith(".xlsx"):

                self.acf_table.to_excel(
                    file_path,
                    index=False
                )

            else:

                self.acf_table.to_csv(
                    file_path,
                    index=False
                )

            messagebox.showinfo(
                "Saved",
                f"Table exported successfully:\n{file_path}"
            )

        except Exception as error:

            messagebox.showerror(
                "Export Error",
                str(error)
            )


# =========================================================
# Run Application
# =========================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = ARIMAGUI(root)

    root.mainloop()
