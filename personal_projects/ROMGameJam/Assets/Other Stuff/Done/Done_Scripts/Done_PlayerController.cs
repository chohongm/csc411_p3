using UnityEngine;
using System.Collections;

[System.Serializable]
public class Done_Boundary 
{
	public float xMin, xMax, zMin, zMax;
}

public class Done_PlayerController : MonoBehaviour {
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

    void Start() {
        rotationSpeed = 55f;
        //speed = 2f;              
    }

    void Update() {

        if (Input.GetKey("space") && Time.time > nextFire) {
            nextFire = Time.time + fireRate;
            Instantiate(shot, shotSpawn.position, shotSpawn.rotation);
            GetComponent<AudioSource>().Play();
        }
        /*
        if (Input.GetKey("w") && !Input.GetKey("a") && !Input.GetKey("d")) {
            GetComponent<Rigidbody>().velocity = new Vector3(0, 0, 1) * speed;
            rot = Quaternion.Euler(0, 0, 0);
        }
        if (Input.GetKey("w") && Input.GetKey("a")) {
            GetComponent<Rigidbody>().velocity = new Vector3(-1, 0, 1) * speed;
            rot = Quaternion.Euler(0, -45, 0);
        }
        if (Input.GetKey("w") && Input.GetKey("d")) {
            GetComponent<Rigidbody>().velocity = new Vector3(1, 0, 1) * speed;
            rot = Quaternion.Euler(0, 45, 0);
        }
        if (Input.GetKey("s") && !Input.GetKey("a") && !Input.GetKey("d")) {
            GetComponent<Rigidbody>().velocity = new Vector3(0, 0, -1) * speed;
            rot = Quaternion.Euler(0, -180, 0);
        }
        if (Input.GetKey("s") && Input.GetKey("a")) {
            GetComponent<Rigidbody>().velocity = new Vector3(-1, 0, -1) * speed;
            rot = Quaternion.Euler(0, -135, 0);
        }
        if (Input.GetKey("s") && Input.GetKey("d")) {
            GetComponent<Rigidbody>().velocity = new Vector3(1, 0, -1) * speed;
            rot = Quaternion.Euler(0, 135, 0);
        }
        if (Input.GetKey("a") && !Input.GetKey("w") && !Input.GetKey("s")) {
            GetComponent<Rigidbody>().velocity = new Vector3(-1, 0, 0) * speed;
            rot = Quaternion.Euler(0, -90, 0);
        }
        if (Input.GetKey("d") && !Input.GetKey("w") && !Input.GetKey("s")) {
            GetComponent<Rigidbody>().velocity = new Vector3(1, 0, 0) * speed;
            rot = Quaternion.Euler(0, 90, 0);
        }

        transform.rotation = Quaternion.RotateTowards(transform.rotation, rot, rotationSpeed * Time.deltaTime);
        */
    }


    void FixedUpdate() {
        rot = transform.rotation;
        // Grab the Z euler angle
        float y = rot.eulerAngles.y;

        // Change the Z angle based on input
        y -= Input.GetAxis("Horizontal") * rotationSpeed * Time.deltaTime;

        // Recreate the quaternion
        rot = Quaternion.Euler(0, y, 0);

        // Feed the quaternion into our rotation
        transform.rotation = rot;

        float moveHorizontal = Input.GetAxis("Horizontal");
        float moveVertical = Input.GetAxis("Vertical");

        Vector3 movement = new Vector3(moveHorizontal, 0.0f, moveVertical);

        Quaternion look;
        // update rotation only when there is movement to prevent not resetting to facing north when no movement.
        if (movement.sqrMagnitude > 0) {
            look = Quaternion.LookRotation(movement);
            GetComponent<Rigidbody>().rotation = Quaternion.RotateTowards(transform.rotation, look, 360.0f * Time.deltaTime);
            //(0.0f, GetComponent<Rigidbody>().velocity.x * -tilt, 0.0f);
            float angleDiff = 100 * Mathf.Abs(Mathf.Abs(GetComponent<Rigidbody>().rotation.y % 360f) - Mathf.Abs(look.y % 360f));

            // movement is allowed only when facing the desired direction within 10 degrees.
            // if not within range, rotate only.
            if (angleDiff <= 10f) {
                GetComponent<Rigidbody>().velocity = movement * speed;
                GetComponent<Rigidbody>().position = new Vector3
                (
                Mathf.Clamp(GetComponent<Rigidbody>().position.x, boundary.xMin, boundary.xMax),
                0.0f,
                Mathf.Clamp(GetComponent<Rigidbody>().position.z, boundary.zMin, boundary.zMax)
                );
            }
        // if no movement, just update speed to halt.
        } else {
            GetComponent<Rigidbody>().velocity = movement * speed;
        }
    }

}
