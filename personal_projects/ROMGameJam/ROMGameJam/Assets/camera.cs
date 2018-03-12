using UnityEngine;
using System.Collections;

public class camera : MonoBehaviour {

    public Transform target;

    // Update is called once per frame
    void FixedUpdate() {
        Vector2 pos = Vector2.Lerp((Vector2)transform.position, (Vector2)target.transform.position, Time.fixedDeltaTime);
        transform.position = new Vector3(pos.x, pos.y, transform.position.z);
    }
}
