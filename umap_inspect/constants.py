from enum import Enum, StrEnum

from fi_misc.data import ImageLabels


class UmapMode(StrEnum):
    RAW = ("raw pixels",)
    PLAIN = ("embedded space",)


IMAGE_LINKS = """<style>
.top-left {
    position: absolute;
    bottom: 8px;
    left: 16px;
}
.gallery {
      --s: 400px; /* control the size */
      display: grid;
      gap: 10px; /* control the gap */
      grid: auto-flow var(--s)/repeat(3,var(--s));
      place-items: center;
      margin: calc(var(--s)/4);
}
.gallery > img {
  width: 100%;
  aspect-ratio: 1;
}
</style><div class="gallery">"""


def generate_tooltips(column_names, search_column):
    tooltips = "<div>"
    if ImageLabels.IMAGE_URL in column_names and ImageLabels.FILENAME in column_names:
        tooltips += """
    <div>
        <img
            src="@image_url" height="150" alt="@filename" width="150"
            style="float: left; margin: 0px 15px 15px 0px;"
            border="2"
        ></img>
    </div>
    <div>
        <span style="font-size: 17px; font-weight: bold;">@filename</span>
    </div>
        """
    if not search_column == ImageLabels.DEFAULT:
        tooltips += """
            <div>
                <span style="font-size: 15px;">{}</span>
                <span style="font-size: 15px;">@{}</span>
            </div>
        """.format(
            search_column, search_column
        )
    tooltips += """
        <div>
            <span style="font-size: 15px;">Location</span>
            <span style="font-size: 10px; color: #696;">($x, $y)</span>
        </div>
        </div>
    """
    return tooltips


TOOLTIPS = """
    <div>
        <div>
            <img
                src="@image_url" height="150" alt="@filename" width="150"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
        <div>
            <span style="font-size: 17px; font-weight: bold;">@filename</span>
        </div>
        ###BLOCK
        <div>
            <span style="font-size: 15px;">Location</span>
            <span style="font-size: 10px; color: #696;">($x, $y)</span>
        </div>
    </div>
"""

METRIC_HELPTEXT = (
    f"KNN: "
    + """
        "fraction of k-nearest neighbours in the original high- dimensional data that are preserved as k-nearest neighbours in the embedding. [...]
        KNN quantifies preservation of the local, or microscopic structure" (Kobak and Berens, 2019).
        <br />
        A value of 1 has kept the information from high-dimensional space perfectly, whereas a value of 0 has not. 
        We use k=10 for this metric.
        <br/>
        <!-- KNC: "K-Nearest Cluster" is "the fraction of k-nearest class means in the original data that are preserved as k-nearest class means in the embedding. This is computed for class means only and averaged across all classes." (Kobak and Berens, 2019).
        In other words, KNC captures mesoscopic structures. It looks at how data points in one cluster is adjacent points in others between high- and low-dimensional space.
        A value of 1 means that class information is preserved perfectly, whereas a value of 0 means that it is not.
        We use k=10 for this metric.
        <br />
        -->
        CPD: "correlation between pairwise distances in the high-dimensional space and in the embedding" (Kobak and Berens, 2019). 
        This is computed for a random subset of 1000 points. 
        CPD is a measure of the global structure of the data.
        A value of 1 means that the distances are perfectly preserved, whereas a value of 0 means that they are not.
        <br />
        SS: Silhouette score (Rousseeuw, 1987) measures how similar an object is to its own cluster compared to other clusters. 
        Silhouette scores are from -1 to 1, where a high value (1) indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.
        Silhoutte scores are best for convex clusterings.
        <br />
        """
)
METRIC_CITETEXT = """<a href="https://www.nature.com/articles/s41467-019-13056-x">Kobak and Berens, 2019</a>
        <br />
        <a href="https://www.sciencedirect.com/science/article/pii/0377042787901257?via%3Dihub">Rousseeuw, 1987</a>
        """
