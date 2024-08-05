private Node<K,V> lookup(final Object data, final int index) {
    <mask>
    Node<K,V> node = rootNode[index];
    while(node != null) {
        int cmp = compare(Node.NO_CHANGE, data, node.getStatus(), node.getData(index), index);
        if(cmp == 0) {
            rval = node;
            break;
        } else {
            node =(cmp < 0) ? node.getLeft(index) : node.getRight(index);
        }
    }
    return rval;
}