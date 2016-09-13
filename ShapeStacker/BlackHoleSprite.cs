using UnityEngine;
using System.Collections;

public class BlackHoleSprite : MonoBehaviour {

    Vector3 scale;
    float newScale;

    // Update is called once per frame
    void FixedUpdate () {
        newScale = GameObject.FindObjectOfType<BlackHole>().collider.radius * 2;
        scale = new Vector3(newScale, newScale, newScale);
        transform.localScale = scale;
    }
}
