using UnityEngine;
using System.Collections;


public class InstantiateOnTerrain : MonoBehaviour {

    public LayerMask mask = -1;
    float radius;

    void Start() {
        // set the vertical offset to the object's collider bounds' extends
        if (GetComponent<Collider>() != null) {
            radius = GetComponent<Collider>().bounds.extents.y;
        } else {
            radius = 1f;
        }

        // raycast to find the y-position of the masked collider at the transforms x/z
        RaycastHit hit;
        // note that the ray starts at 100 units
        Ray ray = new Ray(transform.position + Vector3.up * 100, Vector3.down);

        if (Physics.Raycast(ray, out hit, Mathf.Infinity, mask)) {
            if (hit.collider != null) {
                // this is where the gameobject is actually put on the ground
                transform.position = new Vector3(transform.position.x, hit.point.y + radius, transform.position.z);
            }
        }
    }
}
