# Overpass OSM API has been amended to require additional Header data.
# This module sub-classes the Overpass module to add minimal extension without altering the public class.
#
import overpy
from urllib.request import Request, urlopen
from urllib.error import HTTPError


class OverpassWithHeaders(overpy.Overpass):
    def query(self, query):
        if not isinstance(query, bytes):
            query = query.encode("utf-8")

        req = Request(
            self.url,
            data=query,
            headers={
                "User-Agent": "ChadsTool/1.0 (contact: hiblet@yahoo.com)",
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
            },
            method="POST",
        )

        response = b""

        try:
            with urlopen(req) as f:
                while True:
                    chunk = f.read(self.read_chunk_size)
                    if not chunk:
                        break
                    response += chunk

                content_type = f.getheader("Content-Type") or ""

        except HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Overpass HTTP {exc.code}. Body: {err_body[:2000]}"
            ) from exc

        if "json" in content_type.lower():
            return self.parse_json(response)

        return self.parse_xml(response)