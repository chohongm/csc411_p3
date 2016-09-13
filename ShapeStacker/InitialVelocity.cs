using UnityEngine;
using System.Collections;

public class InitialVelocity : MonoBehaviour {

    public Vector3 launchVelocity;

	// Use this for initialization
	void Start () {
        var body = GetComponent<Rigidbody2D>();
        body.velocity = launchVelocity;
    }
	
	// Update is called once per frame
	void Update () {
	
	}
}
