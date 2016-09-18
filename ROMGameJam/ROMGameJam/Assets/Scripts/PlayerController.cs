using UnityEngine;
using System.Collections;

[System.Serializable]
public class Done_Boundary {
    public float xMin, xMax, yMin, yMax;
}

public class PlayerController : MonoBehaviour {
    private GameController gameController;
    private static float MIN_SPEED = 1.0f; // speed player must pass to play the animation

    public float speed;
    public float tilt;
    public Done_Boundary boundary;

    public GameObject shot;
    public Transform shotSpawn;
    public float fireRate;

    private float nextFire;
    // Grab our rotation quaternion
    private Quaternion rot;
    public float rotationSpeed;
    public Transform offsetBG;
    public Transform camera;
    private Rigidbody2D rb;


    void Start() {
        rotationSpeed = 55f;
        //speed = 2f;    
        rb = transform.GetComponent<Rigidbody2D>();
        gameController = GameObject.FindWithTag("GameController").GetComponent<GameController>();
    }

    void Update() {
        if (gameController.GetCreatureSelected()) {
            float moveHorizontal = Input.GetAxis("Horizontal");
            float moveVertical = Input.GetAxis("Vertical");

            Vector3 movement = new Vector3(moveHorizontal, moveVertical, 0f);



            if (moveHorizontal != 0 || moveVertical != 0) {
                float angle = Mathf.Atan2(-moveHorizontal, moveVertical) * Mathf.Rad2Deg;
                float turnSpeed = 20;
                transform.rotation = Quaternion.Slerp(transform.rotation, Quaternion.Euler(0, 0, angle), turnSpeed * Time.deltaTime);
                if (angle < 0) {
                    angle += 360;
                }
                float angleDiff = Mathf.Abs(Mathf.Abs(transform.rotation.eulerAngles.z) - Mathf.Abs(angle));
                //Debug.Log(movement);

                // movement is allowed only when facing the desired direction within 10 degrees.
                // if not within range, rotate only.
                if (angleDiff <= 20f) {
                    rb.velocity = movement * speed;
                    //offsetBG.GetComponent<Rigidbody>().velocity = movement * speed;
                    //camera.GetComponent<Rigidbody>().velocity = movement * speed;


                }
            } else {
                rb.velocity = movement * speed;
                //offsetBG.GetComponent<Rigidbody>().velocity = movement * speed;
                //camera.GetComponent<Rigidbody>().velocity = movement * speed;

            }
            if (Input.GetKeyDown(KeyCode.Space)) {
                GetComponent<Animator>().SetTrigger("attack");
            }

            // Only play the animation if the player is moving above a specific speed 
            if (Mathf.Abs(rb.velocity.x) < MIN_SPEED && Mathf.Abs(rb.velocity.y) < MIN_SPEED) {
                GetComponent<Animator>().SetBool("movement", false);
            } else {
                GetComponent<Animator>().SetBool("movement", true);
            }
        }

    }
}