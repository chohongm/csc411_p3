using UnityEngine;
using System.Collections;

public class MouseManager : MonoBehaviour
{
    public bool useSpring = false;

    public LineRenderer dragLine;

    //float dragSpeed = 4f;
    float velocityRatio = 4f; // If we aren't using a spring

    Rigidbody2D grabbedObject = null;
    SpringJoint2D springJoint = null;

    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            // We clicked, but on what?
            Vector3 mouseWorldPos3D = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            Vector2 mousePos2D = new Vector2(mouseWorldPos3D.x, mouseWorldPos3D.y);

            Vector2 dir = Vector2.zero;

            RaycastHit2D hit = Physics2D.Raycast(mousePos2D, dir);
            if (hit.collider != null)
            {
                // We clicked on SOMETHING that has a collider
                if (hit.collider.GetComponent<Rigidbody2D>())
                {
                    grabbedObject = hit.collider.GetComponent<Rigidbody2D>();
                 

                    if (useSpring)
                    {
                        springJoint = grabbedObject.gameObject.AddComponent<SpringJoint2D>();
                        grabbedObject.gravityScale = 0;
                        // Set the anchor to the spot on the object that we clicked.
                        Vector3 localHitPoint = grabbedObject.transform.InverseTransformPoint(hit.point);
                        springJoint.anchor = localHitPoint;

                        springJoint.connectedAnchor = mouseWorldPos3D;
                        springJoint.distance = 0f;
                        springJoint.dampingRatio = 0.1f;
                        springJoint.frequency = 1;
                        springJoint.autoConfigureDistance = false;

                        // Enable this if you want to collide with objects still (and you probably do)
                        // This will also WAKE UP the spring.
                        springJoint.enableCollision = true;

                        // This will also WAKE UP the spring, even if it's a totally
                        // redundant line because the connectedBody should already be null
                        springJoint.connectedBody = null;
                    } else
                    { // using velocity instead.
                        grabbedObject.gravityScale = 0;
                    }

                    dragLine.enabled = true;
                }
            }
        }

        if (Input.GetMouseButtonUp(0) && grabbedObject != null)
        {
            if (useSpring)
            {
                Destroy(springJoint);
                springJoint = null;
            } else
            {
                grabbedObject.gravityScale = 0;
            }
            grabbedObject = null;
            dragLine.enabled = false;
        }

        
        if (springJoint)
        {
            Vector3 mouseWorldPos3D = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            springJoint.connectedAnchor = mouseWorldPos3D;
        }
        

    }


    void FixedUpdate()
    {
        if (grabbedObject)
        {
            Vector2 mouseWorldPos2D = Camera.main.ScreenToWorldPoint(Input.mousePosition);
            if (useSpring)
            {
                springJoint.connectedAnchor = mouseWorldPos2D;
            } else
            {
                grabbedObject.velocity = (mouseWorldPos2D - grabbedObject.position) * velocityRatio;
            }
        }
    }

    void LateUpdate()
    {
        if (grabbedObject)
        {
            if (useSpring)
            {
                Vector3 worldAnchor = grabbedObject.transform.TransformPoint(springJoint.anchor);
                dragLine.SetPosition(0, new Vector3(worldAnchor.x, worldAnchor.y, -1));
                dragLine.SetPosition(1, new Vector3(springJoint.connectedAnchor.x, springJoint.connectedAnchor.y, -1));
            } else
            {
                Vector3 mouseWorldPos3D = Camera.main.ScreenToWorldPoint(Input.mousePosition);
                dragLine.SetPosition(0, new Vector3(grabbedObject.position.x, grabbedObject.position.y, -1));
                dragLine.SetPosition(1, new Vector3(mouseWorldPos3D.x, mouseWorldPos3D.y, -1));
            }
        }
    }

}
