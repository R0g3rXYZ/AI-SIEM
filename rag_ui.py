import threading
import traceback
import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from dateutil import parser as dtparser

from rag_query import RAGEngine


DEFAULT_IP = "202.93.142.22"
DEFAULT_DATE = "2026-02-09"
DEFAULT_QUESTION = "What type of attack is this and how was it handled by the server?"

SEARCH_MODES = ["By Source IP", "By Date"]

ATTACK_TYPES = [
    "Any",
    "scanner/probing",
    "sql_injection",
    "xss",
    "rce_attempt",
    "path_traversal",
    "http_header_injection",
    "protocol_smuggling_or_crlf",
    "auth_bypass_attempt",
    "auth_bruteforce_or_guessing",
    "cms_probe",
    "unknown",
]


class SimpleRAGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI-SIEM Severity Viewer")
        self.root.geometry("980x700")
        self.root.configure(bg="#f4efe7")

        self.engine = None
        self.is_running = False

        self.mode_var = tk.StringVar(value=SEARCH_MODES[0])
        self.input_var = tk.StringVar(value=DEFAULT_IP)
        self.type_filter_var = tk.StringVar(value=ATTACK_TYPES[0])
        self.include_baseline_var = tk.BooleanVar(value=True)
        self.include_raw_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")

        self._build()
        self.mode_var.trace_add("write", self._on_mode_change)

    def _on_mode_change(self, *_):
        mode = self.mode_var.get()
        if mode == "By Source IP":
            self.input_label.configure(text="Source IP")
            self.input_var.set(DEFAULT_IP)
        else:
            self.input_label.configure(text="Date (YYYY-MM-DD or 'March 15')")
            self.input_var.set(DEFAULT_DATE)

    def _build(self) -> None:
        # --- Header ---
        header = tk.Frame(self.root, bg="#1f3a5f", padx=18, pady=18)
        header.pack(fill="x", padx=16, pady=(16, 10))

        tk.Label(
            header,
            text="AI-SIEM Severity Viewer",
            font=("Segoe UI", 18, "bold"),
            fg="#f8fafc",
            bg="#1f3a5f",
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Review the most severe finding for a source IP or date and inspect the generated triage summary.",
            font=("Segoe UI", 10),
            fg="#d8e4f1",
            bg="#1f3a5f",
        ).pack(anchor="w", pady=(6, 0))

        # --- Controls ---
        controls = tk.Frame(self.root, bg="#fffaf2", bd=1, relief="solid", padx=16, pady=14)
        controls.pack(fill="x", padx=16)

        tk.Label(
            controls,
            text="Search",
            font=("Segoe UI", 12, "bold"),
            fg="#243b53",
            bg="#fffaf2",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # Row 1: column labels
        tk.Label(controls, text="Search Mode", fg="#486581", bg="#fffaf2").grid(
            row=1, column=0, sticky="w"
        )
        tk.Label(controls, text="Attack Type Filter", fg="#486581", bg="#fffaf2").grid(
            row=1, column=1, sticky="w", padx=(12, 0)
        )

        # Row 2: dropdowns side by side
        mode_menu = tk.OptionMenu(controls, self.mode_var, *SEARCH_MODES)
        mode_menu.configure(
            bg="#ffffff", fg="#102a43", relief="flat",
            highlightthickness=1, highlightbackground="#bcccdc",
            activebackground="#e9eef5", font=("Segoe UI", 10),
        )
        mode_menu["menu"].configure(bg="#ffffff", fg="#102a43", font=("Segoe UI", 10))
        mode_menu.grid(row=2, column=0, sticky="ew", pady=(4, 10))

        type_menu = tk.OptionMenu(controls, self.type_filter_var, *ATTACK_TYPES)
        type_menu.configure(
            bg="#ffffff", fg="#102a43", relief="flat",
            highlightthickness=1, highlightbackground="#bcccdc",
            activebackground="#e9eef5", font=("Segoe UI", 10),
        )
        type_menu["menu"].configure(bg="#ffffff", fg="#102a43", font=("Segoe UI", 10))
        type_menu.grid(row=2, column=1, sticky="ew", padx=(12, 0), pady=(4, 10))

        # Row 3: dynamic input label
        self.input_label = tk.Label(controls, text="Source IP", fg="#486581", bg="#fffaf2")
        self.input_label.grid(row=3, column=0, columnspan=2, sticky="w")

        # Row 4: input entry
        self.input_entry = tk.Entry(
            controls,
            textvariable=self.input_var,
            width=24,
            bg="#ffffff",
            fg="#102a43",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#bcccdc",
            highlightcolor="#d64545",
            font=("Segoe UI", 10),
        )
        self.input_entry.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 12))

        # Row 5: checkboxes
        options = tk.Frame(controls, bg="#fffaf2")
        options.grid(row=5, column=0, columnspan=2, sticky="w")

        self.baseline_check = tk.Checkbutton(
            options,
            text="Include same-day counts and baseline stats",
            variable=self.include_baseline_var,
            bg="#fffaf2",
            fg="#243b53",
            activebackground="#fffaf2",
            selectcolor="#fffaf2",
        )
        self.baseline_check.pack(side="left", padx=(0, 20))

        self.raw_check = tk.Checkbutton(
            options,
            text="Include raw log preview",
            variable=self.include_raw_var,
            bg="#fffaf2",
            fg="#243b53",
            activebackground="#fffaf2",
            selectcolor="#fffaf2",
        )
        self.raw_check.pack(side="left")

        # Row 6: buttons
        actions = tk.Frame(controls, bg="#fffaf2")
        actions.grid(row=6, column=0, columnspan=2, sticky="w", pady=(14, 0))

        self.run_button = tk.Button(
            actions,
            text="Run",
            width=12,
            command=self.run_query,
            bg="#d64545",
            fg="#ffffff",
            relief="flat",
            activebackground="#b83737",
            activeforeground="#ffffff",
            padx=8,
            pady=6,
        )
        self.run_button.pack(side="left")

        self.clear_button = tk.Button(
            actions,
            text="Clear",
            width=12,
            command=self.clear_output,
            bg="#e9eef5",
            fg="#243b53",
            relief="flat",
            activebackground="#d9e2ec",
            activeforeground="#102a43",
            padx=8,
            pady=6,
        )
        self.clear_button.pack(side="left", padx=(10, 0))

        controls.grid_columnconfigure(0, weight=1)
        controls.grid_columnconfigure(1, weight=1)

        # --- Status ---
        status_frame = tk.Frame(self.root, bg="#f4efe7", padx=18, pady=10)
        status_frame.pack(fill="x")

        tk.Label(
            status_frame,
            textvariable=self.status_var,
            anchor="w",
            font=("Segoe UI", 10, "bold"),
            fg="#9f1239",
            bg="#f4efe7",
        ).pack(fill="x")

        # --- Results ---
        results = tk.Frame(self.root, bg="#ffffff", bd=1, relief="solid", padx=6, pady=6)
        results.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self.output = ScrolledText(
            results,
            wrap="word",
            padx=16,
            pady=16,
            bg="#ffffff",
            fg="#102a43",
            insertbackground="#102a43",
            relief="flat",
            font=("Segoe UI", 10),
        )
        self.output.pack(fill="both", expand=True)
        self.output.configure(state="disabled")
        self._configure_output_tags()

    def set_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self._insert_formatted_text(text)
        self.output.configure(state="disabled")

    def clear_output(self) -> None:
        if self.is_running:
            return
        self.mode_var.set(SEARCH_MODES[0])
        self.input_var.set(DEFAULT_IP)
        self.type_filter_var.set(ATTACK_TYPES[0])
        self.include_baseline_var.set(True)
        self.include_raw_var.set(False)
        self.set_output("")
        self.status_var.set("Ready")

    def set_running(self, running: bool, status: str) -> None:
        self.is_running = running
        state = "disabled" if running else "normal"
        self.run_button.configure(state=state)
        self.clear_button.configure(state=state)
        self.input_entry.configure(state=state)
        self.baseline_check.configure(state=state)
        self.raw_check.configure(state=state)
        self.status_var.set(status)

    def _get_engine(self) -> RAGEngine:
        if self.engine is None:
            self.engine = RAGEngine()
        return self.engine

    def run_query(self) -> None:
        if self.is_running:
            return

        mode = self.mode_var.get()
        input_val = self.input_var.get().strip()
        if not input_val:
            input_val = DEFAULT_IP if mode == "By Source IP" else DEFAULT_DATE

        type_filter = self.type_filter_var.get()
        if type_filter == "Any":
            type_filter = None

        include_baseline = self.include_baseline_var.get()
        include_raw = self.include_raw_var.get()

        self.set_running(True, "Loading models and querying logs...")
        self.set_output("")

        worker = threading.Thread(
            target=self._run_query_worker,
            args=(mode, input_val, DEFAULT_QUESTION, include_baseline, include_raw, type_filter),
            daemon=True,
        )
        worker.start()

    def _run_query_worker(
        self, mode: str, input_val: str, question: str,
        include_baseline: bool, include_raw: bool, type_filter
    ) -> None:
        try:
            engine = self._get_engine()

            if mode == "By Source IP":
                result = engine.workflow_most_severe_for_ip(input_val, attack_type_filter=type_filter)
            else:
                result = engine.workflow_most_severe_on_day(input_val, attack_type_filter=type_filter)

            if "error" in result:
                self.root.after(0, lambda: self._finish_with_text(result["error"]))
                return

            best = result["best_chunk"]
            explanation = engine.answer_with_context(question, [best])

            day_stats = None
            baseline = None
            if include_baseline:
                try:
                    day_dt = dtparser.parse(best.get("time_start", ""))
                    day_stats = engine.workflow_count_type_on_day(
                        result["attack_type"],
                        day_dt.replace(hour=0, minute=0, second=0, microsecond=0),
                    )
                except Exception:
                    day_stats = None
                baseline = engine.workflow_commonness(result["attack_type"])

            output = self._format_result(result, mode, explanation, day_stats, baseline, include_raw)
            self.root.after(0, lambda: self._finish_with_text(output))
        except Exception as exc:
            error_text = f"{exc}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: self._finish_with_error(error_text))

    def _finish_with_text(self, text: str) -> None:
        self.set_output(text)
        self.set_running(False, "Done")

    def _finish_with_error(self, error_text: str) -> None:
        self.set_output(error_text)
        self.set_running(False, "Error")
        messagebox.showerror("AI-SIEM Severity Viewer", "The query could not be completed. Check the output for details.")

    def _configure_output_tags(self) -> None:
        self.output.tag_configure("title", font=("Segoe UI", 18, "bold"), foreground="#9f1239", spacing3=10)
        self.output.tag_configure("section", font=("Segoe UI", 12, "bold"), foreground="#1f3a5f", spacing1=12, spacing3=6)
        self.output.tag_configure("label", font=("Segoe UI", 10, "bold"), foreground="#243b53")
        self.output.tag_configure("value", font=("Segoe UI", 10), foreground="#102a43")
        self.output.tag_configure("muted", font=("Segoe UI", 10), foreground="#627d98")
        self.output.tag_configure("body", font=("Segoe UI", 10), foreground="#102a43", spacing3=6)
        self.output.tag_configure("code", font=("Consolas", 10), foreground="#7c2d12", background="#fff7ed", lmargin1=8, lmargin2=8, spacing3=4)

    def _insert_formatted_text(self, text: str) -> None:
        if not text:
            return

        for raw_line in text.splitlines():
            line = raw_line.rstrip("\n")
            if line.startswith("## "):
                self.output.insert("end", line[3:] + "\n", ("section",))
            elif line.startswith("# "):
                self.output.insert("end", line[2:] + "\n", ("title",))
            elif line.startswith("```"):
                continue
            elif line.startswith("- "):
                self.output.insert("end", u"\u2022 " + line[2:] + "\n", ("body",))
            elif ": " in line and not line.startswith("http"):
                key, value = line.split(": ", 1)
                self.output.insert("end", key + ": ", ("label",))
                self.output.insert("end", value + "\n", ("value",))
            elif line.startswith("    "):
                self.output.insert("end", line.strip() + "\n", ("code",))
            elif not line:
                self.output.insert("end", "\n", ("body",))
            else:
                self.output.insert("end", line + "\n", ("body",))

    def _format_result(
        self, result, mode: str, explanation: str, day_stats, baseline, include_raw: bool
    ) -> str:
        best = result["best_chunk"]

        if mode == "By Source IP":
            primary_line = f"IP: {result.get('ip', '')}"
        else:
            primary_line = f"Date: {result.get('day', '')}"

        lines = [
            "# Most Severe Finding",
            "",
            "## Overview",
            primary_line,
            f"Chunk ID: {result.get('best_chunk_id', '')}",
            f"Attack Type: {result.get('attack_type', '')}",
            f"Candidate Count: {result.get('candidate_count', 0)}",
            f"Time Start: {best.get('time_start', '')}",
            f"Time End: {best.get('time_end', '')}",
            f"Website: {best.get('website', '') or 'Unknown'}",
            "",
            "## Summary",
            f"{best.get('summary', '')}",
            "",
            "## Request Paths",
        ]

        paths = best.get("paths", []) or []
        if paths:
            for path in paths[:8]:
                lines.append(f"- {path}")
        else:
            lines.append("No paths found.")

        lines.extend([
            "",
            "## Explanation",
            explanation.strip(),
        ])

        if day_stats:
            lines.extend([
                "",
                "## Same-Day Count",
                f"Day: {day_stats.get('day', '')}",
                f"Attack Type Count: {day_stats.get('count', 0)}",
                f"Chunks Seen That Day: {day_stats.get('chunks_seen', 0)}",
            ])

        if baseline and "note" not in baseline:
            lines.extend([
                "",
                "## Baseline",
                f"Days Observed: {baseline.get('days_observed', 0)}",
                f"Average Per Day: {baseline.get('average_per_day', 0):.2f}",
                f"Peak Day: {baseline.get('max_day', '')}",
                f"Peak Day Count: {baseline.get('max_count', 0)}",
            ])
        elif baseline and baseline.get("note"):
            lines.extend(["", "## Baseline", baseline["note"]])

        if include_raw:
            raw_text = (best.get("text", "") or "").strip()
            preview = raw_text[:1200] if raw_text else "No raw text found."
            lines.extend([
                "",
                "## Raw Log Preview",
                f"    {preview}",
            ])

        return "\n".join(lines)


def main() -> None:
    root = tk.Tk()
    app = SimpleRAGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
