// cc-statusline — compact single-line Claude Code statusline (16-colour)
//
// Reads the statusline JSON from stdin and prints one line.
// Usage: cc-statusline < statusline.json
//
// Never forks: user/host come from the OS API, the git branch is read
// straight from .git/HEAD, and the whole line is written with a single write.
//
// Portable C++20: Linux/BSD/macOS, and native Windows (MSYS2 UCRT64/MINGW64,
// or MSVC). Paths are handled as UTF-8 and go through std::filesystem, so
// non-ASCII and backslash-separated Windows paths work.

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

#ifdef _WIN32
#	ifndef WIN32_LEAN_AND_MEAN
#		define WIN32_LEAN_AND_MEAN
#	endif
#	ifndef NOMINMAX
#		define NOMINMAX
#	endif
#	include <windows.h>
#	include <fcntl.h>
#	include <io.h>
#else
#	include <pwd.h>
#	include <unistd.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

// ── text helpers ─────────────────────────────────────────────────────

std::string_view trim(std::string_view s)
{
	while(!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
		s.remove_suffix(1);
	while(!s.empty() && s.front() == ' ')
		s.remove_prefix(1);
	return s;
}

// UTF-8 <-> fs::path without going through the (Windows ANSI) narrow codepage
fs::path to_path(std::string_view utf8)
{
	return fs::path(std::u8string(utf8.begin(), utf8.end()));
}

std::string from_path(const fs::path &p)
{
	const std::u8string u = p.u8string();
	return std::string(u.begin(), u.end());
}

#ifdef _WIN32
std::string narrow(const wchar_t *w)
{
	int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
	if(n <= 1)
		return {};
	std::string s(static_cast<std::size_t>(n - 1), '\0');
	WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
	return s;
}
#endif

// ── I/O ──────────────────────────────────────────────────────────────

std::string read_stream(std::FILE *f)
{
	std::string s;
	char buf[8192];
	std::size_t n;
	while((n = std::fread(buf, 1, sizeof buf, f)) > 0)
		s.append(buf, n);
	return s;
}

std::optional<std::string> read_file(const fs::path &path)
{
	std::ifstream in(path, std::ios::binary);
	if(!in)
		return std::nullopt;
	return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// ── JSON access ──────────────────────────────────────────────────────

const json *dig(const json &j, std::initializer_list<const char *> path)
{
	const json *cur = &j;
	for(const char *key : path)
	{
		if(!cur->is_object())
			return nullptr;
		auto it = cur->find(key);
		if(it == cur->end() || it->is_null())
			return nullptr;
		cur = &*it;
	}
	return cur;
}

std::string str(const json &j, std::initializer_list<const char *> path)
{
	const json *p = dig(j, path);
	return p && p->is_string() ? p->get<std::string>() : std::string{};
}

std::optional<double> num(const json &j, std::initializer_list<const char *> path)
{
	const json *p = dig(j, path);
	if(p && p->is_number())
		return p->get<double>();
	return std::nullopt;
}

std::optional<int> pct(std::optional<double> v)
{
	if(!v)
		return std::nullopt;
	return static_cast<int>(std::lround(*v));
}

// ── environment (no fork) ────────────────────────────────────────────

std::string user_name()
{
#ifdef _WIN32
	wchar_t buf[257];
	DWORD n = sizeof buf / sizeof *buf;
	if(GetUserNameW(buf, &n))
		return narrow(buf);
	if(const char *u = std::getenv("USERNAME"))
		return u;
#else
	if(const passwd *pw = getpwuid(geteuid()); pw && pw->pw_name)
		return pw->pw_name;
	if(const char *u = std::getenv("USER"))
		return u;
#endif
	return "?";
}

std::string short_host()
{
	std::string h;
#ifdef _WIN32
	wchar_t buf[256];
	DWORD n = sizeof buf / sizeof *buf;
	if(GetComputerNameExW(ComputerNameDnsHostname, buf, &n))
		h = narrow(buf);
	else if(const char *c = std::getenv("COMPUTERNAME"))
		h = c;
#else
	char buf[256] = {};
	if(gethostname(buf, sizeof buf - 1) == 0)
		h = buf;
#endif
	if(auto dot = h.find('.'); dot != std::string::npos)
		h.resize(dot);
	return h.empty() ? "?" : h;
}

std::string current_dir()
{
	std::error_code ec;
	fs::path p = fs::current_path(ec);
	return ec ? std::string{} : from_path(p);
}

std::string home_dir()
{
#ifdef _WIN32
	// _wgetenv: getenv() would hand back the ANSI codepage, not UTF-8
	const wchar_t *h = _wgetenv(L"USERPROFILE");
	return h ? narrow(h) : std::string{};
#else
	const char *h = std::getenv("HOME");
	return h ? h : "";
#endif
}

bool is_sep(char c)
{
#ifdef _WIN32
	return c == '/' || c == '\\';
#else
	return c == '/';
#endif
}

// Windows paths are case-insensitive and accept either separator
bool same_path_char(char a, char b)
{
#ifdef _WIN32
	if(is_sep(a) && is_sep(b))
		return true;
	auto lower = [](char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; };
	return lower(a) == lower(b);
#else
	return a == b;
#endif
}

// /home/neo_chen/vcs -> ~/vcs (only on a whole path component)
std::string tilde(const std::string &path)
{
	std::string home = home_dir();
	while(home.size() > 1 && is_sep(home.back()))
		home.pop_back();
	if(home.empty() || (home.size() == 1 && is_sep(home[0])) || path.size() < home.size())
		return path;
	for(std::size_t i = 0; i < home.size(); ++i)
		if(!same_path_char(path[i], home[i]))
			return path;
	if(path.size() == home.size())
		return "~";
	if(!is_sep(path[home.size()]))
		return path;
	return "~" + path.substr(home.size());
}

// Equivalent of `git symbolic-ref --short HEAD`: walk up looking for .git
// (a directory, or a gitdir: file for worktrees/submodules), read HEAD.
std::string git_branch(const std::string &cwd)
{
	if(cwd.empty())
		return {};

	std::error_code ec;
	for(fs::path dir = to_path(cwd);;)
	{
		const fs::path dotgit = dir / ".git";
		const fs::file_status st = fs::status(dotgit, ec);

		if(fs::is_directory(st) || fs::is_regular_file(st))
		{
			fs::path gitdir = dotgit;
			if(fs::is_regular_file(st))
			{
				auto content = read_file(dotgit);
				if(!content)
					return {};
				std::string_view c = trim(*content);
				constexpr std::string_view tag = "gitdir:";
				if(!c.starts_with(tag))
					return {};
				gitdir = to_path(trim(c.substr(tag.size())));
				if(gitdir.is_relative())
					gitdir = dir / gitdir;
			}

			auto head = read_file(gitdir / "HEAD");
			if(!head)
				return {};
			std::string_view h = trim(*head);
			constexpr std::string_view ref = "ref:";
			if(!h.starts_with(ref))
				return {};	// detached HEAD
			h = trim(h.substr(ref.size()));
			constexpr std::string_view heads = "refs/heads/";
			if(h.starts_with(heads))
				h.remove_prefix(heads.size());
			return std::string{h};
		}

		fs::path parent = dir.parent_path();
		if(parent.empty() || parent == dir)
			return {};
		dir = std::move(parent);
	}
}

// ── colours (16-colour SGR, bright variants) ─────────────────────────

namespace sgr {
constexpr std::string_view
	label   = "90",    // bright black: labels, separators, secondary text
	red     = "91",
	green   = "92",
	yellow  = "93",    // also: session cost
	blue    = "1;94",  // cwd
	magenta = "95",    // model
	cyan    = "96",    // branch
	white   = "97",    // effort
	vim     = "1;95";
}

enum class Level { ok, warn, bad };

// higher is worse (context usage)
Level hi(int n) { return n >= 80 ? Level::bad : n >= 60 ? Level::warn : Level::ok; }
// lower is worse (rate limit remaining)
Level lo(int n) { return n <= 20 ? Level::bad : n <= 40 ? Level::warn : Level::ok; }

std::string_view fg(Level l)
{
	return l == Level::bad ? sgr::red : l == Level::warn ? sgr::yellow : sgr::green;
}

struct Line
{
	std::string s;

	Line &raw(std::string_view t) { s += t; return *this; }
	Line &c(std::string_view code, std::string_view t)
	{
		s += "\033[";
		s += code;
		s += 'm';
		s += t;
		s += "\033[0m";
		return *this;
	}
};

// ── data ─────────────────────────────────────────────────────────────

struct Info
{
	std::string cwd, user, host, branch;
	std::string model, effort, session, vim, cold;
	std::optional<int> ctx, five, week, hit;
	std::optional<double> cost;
	std::optional<long long> added, removed;
};

Info collect(const json &j)
{
	Info in;
	in.cwd = str(j, {"workspace", "current_dir"});
	if(in.cwd.empty())
		in.cwd = str(j, {"cwd"});
	if(in.cwd.empty())
		in.cwd = current_dir();
	in.user = user_name();
	in.host = short_host();
	in.branch = git_branch(in.cwd);

	in.model = str(j, {"model", "display_name"});
	in.effort = str(j, {"effort", "level"});
	in.session = str(j, {"session_name"});
	in.vim = str(j, {"vim", "mode"});

	in.ctx = pct(num(j, {"context_window", "used_percentage"}));
	if(auto u = num(j, {"rate_limits", "five_hour", "used_percentage"}))
		in.five = pct(100 - *u);
	if(auto u = num(j, {"rate_limits", "seven_day", "used_percentage"}))
		in.week = pct(100 - *u);

	const json *observed = dig(j, {"prompt_cache", "caching_observed"});
	const json *warm = dig(j, {"prompt_cache", "warm"});
	if(observed && *observed == true && warm && *warm == false)
	{
		in.cold = "unknown";
		const json *causes = dig(j, {"prompt_cache", "last_miss_cause", "causes"});
		if(causes && causes->is_array() && !causes->empty() && (*causes)[0].is_string())
			in.cold = (*causes)[0].get<std::string>();
	}
	if(auto r = num(j, {"prompt_cache", "hit_ratio"}))
		in.hit = pct(*r * 100);

	in.cost = num(j, {"cost", "total_cost_usd"});
	if(auto a = num(j, {"cost", "total_lines_added"}))
		in.added = std::llround(*a);
	if(auto r = num(j, {"cost", "total_lines_removed"}))
		in.removed = std::llround(*r);
	return in;
}

std::string money(double c)
{
	char buf[32];
	std::snprintf(buf, sizeof buf, "$%.2f", c);
	return buf;
}

std::string percent(int n) { return std::to_string(n) + "%"; }

// ── render ───────────────────────────────────────────────────────────
//
// user@host ~/full/path branch | MODE | Model effort | ctx N% | 5h N% 7d N% | cache N% | $N.NN | +N -N | session

std::string render(const Info &in)
{
	Line l;
	const std::string sep = Line{}.c(sgr::label, " | ").s;

	l.c(sgr::green, in.user + "@" + in.host).raw(" ").c(sgr::blue, tilde(in.cwd));
	if(!in.branch.empty())
		l.raw(" ").c(sgr::cyan, in.branch);
	if(!in.vim.empty())
		l.raw(sep).c(sgr::vim, in.vim);
	if(!in.model.empty())
	{
		l.raw(sep).c(sgr::magenta, in.model);
		if(!in.effort.empty())
			l.raw(" ").c(sgr::white, in.effort);
	}
	if(in.ctx)
		l.raw(sep).c(sgr::label, "ctx ").c(fg(hi(*in.ctx)), percent(*in.ctx));
	if(in.five || in.week)
	{
		l.raw(sep);
		if(in.five)
			l.c(sgr::label, "5h ").c(fg(lo(*in.five)), percent(*in.five));
		if(in.five && in.week)
			l.raw(" ");
		if(in.week)
			l.c(sgr::label, "7d ").c(fg(lo(*in.week)), percent(*in.week));
	}
	if(!in.cold.empty())
		l.raw(sep).c(sgr::label, "cache ").c(sgr::red, "cold " + in.cold);
	else if(in.hit)
		l.raw(sep).c(sgr::label, "cache ").c(sgr::green, percent(*in.hit));
	if(in.cost)
		l.raw(sep).c(sgr::yellow, money(*in.cost));
	if(in.added || in.removed)
	{
		l.raw(sep);
		if(in.added)
			l.c(sgr::green, "+" + std::to_string(*in.added));
		if(in.added && in.removed)
			l.raw(" ");
		if(in.removed)
			l.c(sgr::red, "-" + std::to_string(*in.removed));
	}
	if(!in.session.empty())
		l.raw(sep).c(sgr::label, in.session);

	l.s += '\n';
	return std::move(l.s);
}

} // namespace

int main()
{
#ifdef _WIN32
	// no CRLF translation: Claude Code reads raw bytes
	_setmode(_fileno(stdin), _O_BINARY);
	_setmode(_fileno(stdout), _O_BINARY);
#endif

	json j = json::parse(read_stream(stdin), nullptr, /*allow_exceptions=*/false);
	if(j.is_discarded())
		j = json::object();

	const std::string out = render(collect(j));
	std::fwrite(out.data(), 1, out.size(), stdout);
	return 0;
}
