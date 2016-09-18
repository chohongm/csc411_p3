using UnityEngine;
using System.Collections;

public class Plants : MonoBehaviour {

    public float scrollSpeed;
    private Vector2 savedOffset;
    private Vector3 velocity;
    public GameObject player;
    private Vector3 previousVelocity;
    private float x, y;
    private float x_plustimer, x_minustimer, y_plustimer, y_minustimer;

    void Start() {
        savedOffset = GetComponent<Renderer>().sharedMaterial.GetTextureOffset("_MainTex");
        velocity = player.GetComponent<Rigidbody2D>().velocity;
        previousVelocity = velocity;
        x = transform.position.x;
        y = transform.position.y;
        x_plustimer = 0;
        x_minustimer = 0;
        y_plustimer = 0;
        y_minustimer = 0;
    }

    void FixedUpdate() {
        previousVelocity = velocity;
    }

    void Update() {
        velocity = player.GetComponent<Rigidbody2D>().velocity;

        if (velocity.sqrMagnitude >= previousVelocity.sqrMagnitude && velocity.sqrMagnitude != 0) {
            //Debug.Log("vel: " + velocity.sqrMagnitude);
            //Debug.Log("prev vel: " + previousVelocity.sqrMagnitude);
            if (velocity.x > 0) {
                //x_plustimer += Time.deltaTime;
                //x = Mathf.Repeat(scrollSpeed * x_plustimer, 1);
                x -= scrollSpeed * velocity.x / 10 * Time.deltaTime;
            } else if (velocity.x < 0) {
                //x_plustimer += Time.deltaTime;
                //x = Mathf.Repeat(-scrollSpeed * x_plustimer, 1);
                x -= scrollSpeed * velocity.x / 10 * Time.deltaTime;
            }

            if (velocity.y > 0) {
                //y_plustimer += Time.deltaTime;
                //y = Mathf.Repeat(scrollSpeed * y_plustimer, 1);
                y -= scrollSpeed * velocity.y / 10 * Time.deltaTime;
            } else if (velocity.y < 0) {
                //y_plustimer += Time.deltaTime;
                //y = Mathf.Repeat(scrollSpeed * y_plustimer, 1);
                y -= scrollSpeed * velocity.y / 10 * Time.deltaTime;
            }
            //float x = Mathf.Repeat(velocity.x % 1 * scrollSpeed * Time.time, 1);
            //Debug.Log("x: " + x);
            //float y = Mathf.Repeat(velocity.y % 1 * scrollSpeed * Time.time, 1);
            //Debug.Log("y: " + y);
            //Debug.Log(velocity);
            //Vector2 offset = new Vector2(x, y);
            //GetComponent<Renderer>().sharedMaterial.SetTextureOffset("_MainTex", offset);
            transform.position = new Vector2(x, y);
        }
        //previousVelocity = velocity;
    }

    void OnDisable() {
        GetComponent<Renderer>().sharedMaterial.SetTextureOffset("_MainTex", savedOffset);
    }
}
