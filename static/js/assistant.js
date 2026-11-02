/* The AI assistant panel.

   Reads the server-sent event stream from /api/assistant/chat so tool calls
   appear while the model is still working, rather than a spinner over a
   silent request. */

(function () {
  "use strict";

  var fab = document.getElementById("chat-open");
  var panel = document.getElementById("chat-panel");
  if (!fab || !panel) return;

  var log = document.getElementById("chat-log");
  var form = document.getElementById("chat-form");
  var input = document.getElementById("chat-input");
  var send = document.getElementById("chat-send");
  var statusEl = document.getElementById("chat-status");
  var closeBtn = document.getElementById("chat-close");
  var clearBtn = document.getElementById("chat-clear");

  var history = [];      // [{role, content}] sent back for context
  var busy = false;
  var ready = false;

  // ------------------------------------------------------------- helpers --
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  function scrollDown() { log.scrollTop = log.scrollHeight; }

  function clearIntro() {
    var intro = log.querySelector(".chat-intro");
    if (intro) intro.remove();
  }

  /* Minimal, escaping-first renderer: the model is asked for short prose and
     small tables, so we support just enough Markdown and never inject HTML. */
  function render(target, text) {
    target.textContent = "";
    var lines = String(text).split("\n");
    var i = 0;

    while (i < lines.length) {
      var line = lines[i];

      // A Markdown table: header, separator, then rows.
      if (line.indexOf("|") !== -1 && i + 1 < lines.length &&
          /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[i + 1])) {
        var table = el("table");
        var head = document.createElement("thead");
        var hr = document.createElement("tr");
        splitRow(line).forEach(function (c) { hr.appendChild(el("th", null, c)); });
        head.appendChild(hr);
        table.appendChild(head);

        var body = document.createElement("tbody");
        i += 2;
        while (i < lines.length && lines[i].indexOf("|") !== -1) {
          var tr = document.createElement("tr");
          splitRow(lines[i]).forEach(function (c) {
            tr.appendChild(el("td", null, c));
          });
          body.appendChild(tr);
          i++;
        }
        table.appendChild(body);
        target.appendChild(table);
        continue;
      }

      appendInline(target, line);
      if (i < lines.length - 1) target.appendChild(document.createElement("br"));
      i++;
    }
  }

  function splitRow(line) {
    return line.replace(/^\s*\|/, "").replace(/\|\s*$/, "")
               .split("|").map(function (c) { return c.trim(); });
  }

  // Handles **bold** and `code`; everything else stays literal text.
  function appendInline(target, text) {
    var re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
    var last = 0, m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) {
        target.appendChild(document.createTextNode(text.slice(last, m.index)));
      }
      var tokenText = m[0];
      if (tokenText.charAt(0) === "`") {
        target.appendChild(el("code", null, tokenText.slice(1, -1)));
      } else {
        target.appendChild(el("strong", null, tokenText.slice(2, -2)));
      }
      last = m.index + tokenText.length;
    }
    if (last < text.length) {
      target.appendChild(document.createTextNode(text.slice(last)));
    }
  }

  function addMessage(kind, text) {
    clearIntro();
    var wrap = el("div", "chat-msg " + kind);
    var bubble = el("div", "bubble");
    if (kind === "bot") render(bubble, text); else bubble.textContent = text;
    wrap.appendChild(bubble);
    log.appendChild(wrap);
    scrollDown();
    return wrap;
  }

  function addPending() {
    clearIntro();
    var wrap = el("div", "chat-msg bot");
    var tools = el("div", "chat-tools");
    var bubble = el("div", "bubble");
    var typing = el("span", "chat-typing");
    typing.appendChild(el("span"));
    typing.appendChild(el("span"));
    typing.appendChild(el("span"));
    bubble.appendChild(typing);
    wrap.appendChild(tools);
    wrap.appendChild(bubble);
    log.appendChild(wrap);
    scrollDown();
    return { wrap: wrap, tools: tools, bubble: bubble };
  }

  var TOOL_LABELS = {
    list_tracked_companies: "listed companies",
    get_rankings: "read rankings",
    get_company_detail: "read company",
    get_risk_analysis: "ran risk analysis",
    get_price_history: "read price history",
    search_knowledge: "searched the knowledge base",
    run_sql: "queried the database"
  };

  // --------------------------------------------------------------- state --
  function setStatus(text, bad) {
    statusEl.textContent = text;
    statusEl.classList.toggle("bad", !!bad);
  }

  function setBusy(state) {
    busy = state;
    send.disabled = state || !ready;
    input.disabled = state || !ready;
  }

  /* Only hide the widget if the endpoint itself is unreachable. A missing API
     key is not a reason to hide it: the server falls back to the offline
     engine, which answers from the database and always works. */
  function disable() {
    ready = false;
    fab.remove();
    panel.remove();
  }

  fetch("/api/assistant/status")
    .then(function (r) { return r.json(); })
    .then(function (s) {
      if (!s.ready) { disable(); return; }
      ready = true;
      setStatus(s.mode === "claude" ? s.model : "offline mode", false);
      if (s.mode !== "claude") {
        var note = el("p", null,
          "Running the built-in offline engine — it reads the database " +
          "directly, so no API key is needed. Add an ANTHROPIC_API_KEY for " +
          "free-form questions.");
        note.className = "chat-note";
        var intro = log.querySelector(".chat-intro");
        if (intro) intro.appendChild(note);
      }
      setBusy(false);
    })
    .catch(disable);

  // ---------------------------------------------------------- open/close --
  function open() {
    panel.hidden = false;
    fab.hidden = true;
    fab.setAttribute("aria-expanded", "true");
    if (ready) input.focus();
  }

  function close() {
    panel.hidden = true;
    fab.hidden = false;
    fab.setAttribute("aria-expanded", "false");
    fab.focus();
  }

  fab.addEventListener("click", open);
  closeBtn.addEventListener("click", close);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !panel.hidden) close();
  });

  clearBtn.addEventListener("click", function () {
    history = [];
    log.innerHTML = "";
    var intro = el("div", "chat-intro");
    intro.appendChild(el("p", null, "Conversation cleared. Ask me anything about the data."));
    log.appendChild(intro);
  });

  log.addEventListener("click", function (e) {
    var chip = e.target.closest(".chat-chip");
    if (chip && !busy && ready) {
      input.value = chip.textContent;
      form.dispatchEvent(new Event("submit", { cancelable: true }));
    }
  });

  // Enter sends, Shift+Enter makes a new line.
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.dispatchEvent(new Event("submit", { cancelable: true }));
    }
  });
  input.addEventListener("input", function () {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
  });

  // ----------------------------------------------------------- sending ----
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    if (busy || !ready) return;

    var question = input.value.trim();
    if (!question) return;

    addMessage("user", question);
    input.value = "";
    input.style.height = "auto";
    setBusy(true);

    var pending = addPending();
    var answered = false;

    fetch("/api/assistant/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, history: history })
    })
      .then(function (res) {
        if (!res.ok || !res.body) throw new Error("HTTP " + res.status);

        var reader = res.body.getReader();
        var decoder = new TextDecoder();
        var buffer = "";

        function pump() {
          return reader.read().then(function (chunk) {
            if (chunk.done) return finish();
            buffer += decoder.decode(chunk.value, { stream: true });

            // SSE frames are separated by a blank line.
            var frames = buffer.split("\n\n");
            buffer = frames.pop();
            frames.forEach(function (frame) {
              var line = frame.split("\n").find(function (l) {
                return l.indexOf("data:") === 0;
              });
              if (!line) return;
              var ev;
              try { ev = JSON.parse(line.slice(5).trim()); } catch (err) { return; }
              handle(ev);
            });
            return pump();
          });
        }

        function handle(ev) {
          if (ev.type === "tool") {
            var label = TOOL_LABELS[ev.name] || ev.name;
            pending.tools.appendChild(el("span", "chat-tool", label));
            scrollDown();
          } else if (ev.type === "text" && ev.text) {
            answered = true;
            render(pending.bubble, ev.text);
            history.push({ role: "user", content: question });
            history.push({ role: "assistant", content: ev.text });
            if (history.length > 20) history = history.slice(-20);
            scrollDown();
          } else if (ev.type === "error") {
            answered = true;
            pending.wrap.className = "chat-msg err";
            pending.tools.remove();
            pending.bubble.textContent = ev.message;
            scrollDown();
          }
        }

        function finish() {
          if (!answered) {
            pending.wrap.className = "chat-msg err";
            pending.bubble.textContent = "No answer came back. Try again.";
          }
          setBusy(false);
          input.focus();
        }

        return pump();
      })
      .catch(function (err) {
        pending.wrap.className = "chat-msg err";
        pending.tools.remove();
        pending.bubble.textContent = "Could not reach the assistant: " + err.message;
        setBusy(false);
      });
  });
})();
