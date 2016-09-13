using UnityEngine;
using System.Collections;

class Attracted : MonoBehaviour
{
    string text = ((int)(Random.value * 100)).ToString();

    void Start()
    {
        gameObject.GetComponent<TextMesh>().text = text;
    }

    public GameObject attractedTo;
    public float strengthOfAttraction = 1.0f;

    void FixedUpdate ()
    {
        Vector3 direction = attractedTo.transform.position - transform.position;
        gameObject.GetComponent<Rigidbody2D>().AddForce(strengthOfAttraction * direction);

        //Vector3 relativePos = (attractedTo.transform.position + new Vector3(0, 1.5f, 0)) - transform.position;
        //Quaternion rotation = Quaternion.LookRotation(relativePos);
        //Quaternion current = transform.localRotation;
        //transform.localRotation = Quaternion.Slerp(current, rotation, Time.deltaTime);
        //transform.Translate(0, 0, 3 * Time.deltaTime);

    }
    /*
    float RotationSpeed = 100f;
    float OrbitDegrees = 2f;
    void Update()
    {
        //transform.Rotate(Vector3.up, RotationSpeed * Time.deltaTime);
        transform.RotateAround(attractedTo.transform.position, new Vector3(0, 0, 1), OrbitDegrees);
    }
    */
}