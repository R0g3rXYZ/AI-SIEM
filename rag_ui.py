import threading
import traceback
import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from dateutil import parser as dtparser

from rag_query import RAGEngine


DEFAULT_IP = "202.93.142.22"
DEFAULT_QUESTION = "What type of attack is this and how was it handled by the server?"


class SimpleRAGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI-SIEM Severity Viewer")
        self.root.geometry("980x700")
        self.root.configure(bg="#f4efe7")

        self.engine = None
        self.is_running = False

        self.ip_var = tk.StringVar(value=DEFAULT_IP)
        self.include_baseline_var = tk.BooleanVar(value=True)
        self.include_raw_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")

        self._build()

    def _build(self) -> None:
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
            text="Review the most severe finding for a source IP and inspect the generated triage summary.",
            font=("Segoe UI", 10),
            fg="#d8e4f1",
            bg="#1f3a5f",
        ).pack(anchor="w", pady=(6, 0))

        controls = tk.Frame(self.root, bg="#fffaf2", bd=1, relief="solid", padx=16, pady=14)
        controls.pack(fill="x", padx=16)

        tk.Label(
            controls,
            text="Search",
            font=("Segoe UI", 12, "bold"),
            fg="#243b53",
            bg="#fffaf2",
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        tk.Label(controls, text="Source IP", fg="#486581", bg="#fffaf2").grid(row=1, column=0, sticky="w")
        self.ip_entry = tk.Entry(
            controls,
            textvariable=self.ip_var,
            width=24,
            bg="#ffffff",
            fg="#102a43",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#bcccdc",
            highlightcolor="#d64545",
        )
        self.ip_entry.grid(row=2, column=0, sticky="ew", padx=(0, 12), pady=(4, 12))

        options = tk.Frame(controls, bg="#fffaf2")
        options.grid(row=3, column=0, columnspan=2, sticky="w")

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

        actions = tk.Frame(controls, bg="#fffaf2")
        actions.grid(row=4, column=0, columnspan=2, sticky="w", pady=(14, 0))

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
        controls.grid_columnconfigure(1, weight=3)

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
        self.ip_var.set(DEFAULT_IP)
        self.include_baseline_var.set(True)
        self.include_raw_var.set(False)
        self.set_output("")
        self.status_var.set("Ready")

    def set_running(self, running: bool, status: str) -> None:
        self.is_running = running
        state = "disabled" if running else "normal"
        self.run_button.configure(state=state)
        self.clear_button.configure(state=state)
        self.ip_entry.configure(state=state)
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

        ip = self.ip_var.get().strip() or DEFAULT_IP
        question = DEFAULT_QUESTION
        include_baseline = self.include_baseline_var.get()
        include_raw = self.include_raw_var.get()
        self.set_running(True, "Loading models and querying logs...")
        self.set_output("")

        worker = threading.Thread(
            target=self._run_query_worker,
            args=(ip, question, include_baseline, include_raw),
            daemon=True,
        )
        worker.start()

    def _run_query_worker(self, ip: str, question: str, include_baseline: bool, include_raw: bool) -> None:
        try:
            engine = self._get_engine()
            result = engine.workflow_most_severe_for_ip(ip)

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

            output = self._format_result(result, explanation, day_stats, baseline, include_raw)
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

    def _format_result(self, result, explanation: str, day_stats, baseline, include_raw: bool) -> str:
        best = result["best_chunk"]
        lines = [
            "# Most Severe Finding",
            "",
            "## Overview",
            f"IP: {result.get('ip', '')}",
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

        lines.extend(
            [
                "",
                "## Explanation",
                explanation.strip(),
            ]
        )

        if day_stats:
            lines.extend(
                [
                    "",
                    "## Same-Day Count",
                    f"Day: {day_stats.get('day', '')}",
                    f"Attack Type Count: {day_stats.get('count', 0)}",
                    f"Chunks Seen That Day: {day_stats.get('chunks_seen', 0)}",
                ]
            )

        if baseline and "note" not in baseline:
            lines.extend(
                [
                    "",
                    "## Baseline",
                    f"Days Observed: {baseline.get('days_observed', 0)}",
                    f"Average Per Day: {baseline.get('average_per_day', 0):.2f}",
                    f"Peak Day: {baseline.get('max_day', '')}",
                    f"Peak Day Count: {baseline.get('max_count', 0)}",
                ]
            )
        elif baseline and baseline.get("note"):
            lines.extend(["", "## Baseline", baseline["note"]])

        if include_raw:
            raw_text = (best.get("text", "") or "").strip()
            preview = raw_text[:1200] if raw_text else "No raw text found."
            lines.extend(
                [
                    "",
                    "## Raw Log Preview",
                    f"    {preview}",
                ]
            )

        return "\n".join(lines)


def main() -> None:
    root = tk.Tk()
    app = SimpleRAGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
