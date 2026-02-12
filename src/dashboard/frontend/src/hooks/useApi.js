import { useState, useEffect, useRef } from 'react';

const cache = {};

export function useApi(url) {
    const [data, setData] = useState(cache[url] || null);
    const [loading, setLoading] = useState(!cache[url]);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!url) return;
        if (cache[url]) {
            setData(cache[url]);
            setLoading(false);
            return;
        }

        let cancelled = false;
        setLoading(true);

        fetch(url)
            .then((res) => {
                if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
                return res.json();
            })
            .then((json) => {
                if (!cancelled) {
                    cache[url] = json;
                    setData(json);
                    setLoading(false);
                }
            })
            .catch((err) => {
                if (!cancelled) {
                    setError(err.message);
                    setLoading(false);
                }
            });

        return () => { cancelled = true; };
    }, [url]);

    return { data, loading, error };
}
