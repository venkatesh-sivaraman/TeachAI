
def get_ancestors(widget):
    """Returns a list of ancestors in order such that the given widget appears at the beginning."""
    result = [widget]
    while result[-1].parent is not None:
        if result[-1].parent.parent == result[-1].parent:
            break
        result.append(result[-1].parent)
    return result

def convert_point_to_ancestor(point, descendant, ancestor):
    """Converts a point where descendant is known to be below ancestor in the widget tree."""
    if descendant == ancestor:
        return point
    current = descendant.parent
    result = point
    while current != ancestor:
        result = (result[0] + current.pos[0], result[1] + current.pos[1])
        current = current.parent
    return result

def convert_point_to_descendant(point, ancestor, descendant):
    """Converts a point where ancestor is known to be above descendant in the widget tree."""
    if descendant == ancestor:
        return point
    current = descendant.parent
    parents = [current]
    while current != ancestor:
        current = current.parent
        parents.append(current)
    result = point
    for widget in parents:
        result = (result[0] - widget.pos[0], result[1] - widget.pos[1])
    return result

def convert_point(point, from_widget, to_widget):
    """Converts the given point from the source widget to the destination widget."""

    # Obtain the two widget's ancestor lists and find the most recent common ancestor
    src_ancestors = get_ancestors(from_widget)
    dest_ancestors = get_ancestors(to_widget)
    src_mrca = src_ancestors.pop()
    dest_mrca = dest_ancestors.pop()
    assert dest_mrca == src_mrca, "Widgets do not share a common ancestor!"
    while src_ancestors and dest_ancestors:
        if src_ancestors[-1] != dest_ancestors[-1]:
            break
        src_mrca = src_ancestors.pop()
        dest_mrca = dest_ancestors.pop()

    # Now walk up the tree from from_widget to the MRCA, and back down to to_widget
    mrca_pt = convert_point_to_ancestor(point, from_widget, src_mrca)
    return convert_point_to_descendant(point, src_mrca, to_widget)


