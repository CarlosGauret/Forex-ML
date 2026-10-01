from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[1]


def yfinance_cache_dir(root=None):
    return Path(root or project_root()) / "data" / "yfinance_cache"


def ensure_yfinance_cache(root=None):
    """Configura yfinance para usar un cache SQLite local y escribible."""
    cache_dir = yfinance_cache_dir(root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    probe = cache_dir / ".write_test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)

    try:
        import yfinance.cache as yf_cache

        if hasattr(yf_cache, "set_cache_location"):
            yf_cache.set_cache_location(str(cache_dir))
    except Exception:
        try:
            import yfinance as yf

            if hasattr(yf, "set_tz_cache_location"):
                yf.set_tz_cache_location(str(cache_dir))
        except Exception:
            pass

    return cache_dir
