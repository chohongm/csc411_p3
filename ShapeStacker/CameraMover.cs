using UnityEngine;
using System.Collections;

public class CameraMover : MonoBehaviour {

    public float targetY = 0;
	
	// Update is called once per frame
	void Update () {
        Vector3 pos = transform.position;
        pos.y = Mathf.Lerp(transform.position.y, targetY, 1 * Time.deltaTime);
        transform.position = pos;
	}
}
