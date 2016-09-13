using UnityEngine;
using System.Collections;

public class GridManager : MonoBehaviour {
    public GameObject EmptyHex;
    public GameObject[] LargeTreeHexes;
    public GameObject[] MediumTreeHexes;
    public GameObject[] SmallTreeHexes;

    //This time instead of specifying the number of hexes you should just drop your ground game object on this public variable
    public GameObject Ground;
    public int largeCount;
    public int mediumCount;
    public int smallCount;
    private int instantiatedTreeCount;

    private float hexWidth;
    private float hexHeight;
    private float groundWidth;
    private float groundHeight;

    //public GameObject[] trees;


    void setSizes() {
        hexWidth = EmptyHex.GetComponent<Renderer>().bounds.size.x;
        hexHeight = EmptyHex.GetComponent<Renderer>().bounds.size.z;
        groundWidth = Ground.GetComponent<Renderer>().bounds.size.x;
        groundHeight = Ground.GetComponent<Renderer>().bounds.size.z;
        //Debug.Log("width: " + hexWidth + "\nheight: " + hexHeight + "\ngroundWidth: " + groundWidth + "\ngroundHeight: " + groundHeight);
    }

    //The method used to calculate the number hexagons in a row and number of rows
    //Vector2.x is gridWidthInHexes and Vector2.y is gridHeightInHexes
    Vector2 calcGridSize() {
        //According to the math textbook hexagon's side length is half of the height
        float sideLength = hexHeight / 2;
        //the number of whole hex sides that fit inside inside ground height
        int nrOfSides = (int)(groundHeight / sideLength);
        //I will not try to explain the following calculation because I made some assumptions, which might not be correct in all cases, to come up with the formula. So you'll have to trust me or figure it out yourselves.
        int gridHeightInHexes = (int)(nrOfSides * 2 / 3);
        //When the number of hexes is even the tip of the last hex in the offset column might stick up.
        //The number of hexes in that case is reduced.
        if (gridHeightInHexes % 2 == 0
            && (nrOfSides + 0.5f) * sideLength > groundHeight)
            gridHeightInHexes--;
        //gridWidth in hexes is calculated by simply dividing ground width by hex width
        return new Vector2((int)(groundWidth / hexWidth), gridHeightInHexes);
    }
    //Method to calculate the position of the first hexagon tile
    //The center of the hex grid is (0,0,0)
    Vector3 calcInitPos() {
        Vector3 initPos;
        initPos = new Vector3((-groundWidth / 2 + hexWidth / 2) + Ground.transform.position.x, 0,
            (groundHeight / 2 - hexWidth / 2) + Ground.transform.position.z);

        return initPos;
    }

    Vector3 calcWorldCoord(Vector2 gridPos) {
        Vector3 initPos = calcInitPos();
        float offset = 0;
        if (gridPos.y % 2 != 0)
            offset = hexWidth / 2;

        float x = initPos.x + offset + gridPos.x * hexWidth;
        float z = initPos.z - gridPos.y * hexHeight * 0.75f;
        //If your ground is not a plane but a cube you might set the y coordinate to sth like groundDepth/2 + hexDepth/2
        return new Vector3(x, 30.1f, z);
    }

    void createGrid() {
        Vector2 gridSize = calcGridSize();
        GameObject hexGridGO = new GameObject("HexGrid");
        //hexGridGO.transform.position = Ground.transform.position;

        for (float y = 0; y < gridSize.y; y++) {
            float sizeX = gridSize.x;
            //if the offset row sticks up, reduce the number of hexes in a row
            if (y % 2 != 0 && (gridSize.x + 0.5) * hexWidth > groundWidth)
                sizeX--;
            for (float x = 0; x < sizeX; x++) {
                //GameObject hex = GetTreeHex(); //(GameObject)Instantiate(Hex);
                Vector2 gridPos = new Vector2(x, y);
                Vector3 pos = calcWorldCoord(gridPos);
                Quaternion rot = Quaternion.Euler(0, Random.Range(0, 6) * 60, 0);
                GameObject hex;
                // random rotate the hex tile 60 degrees scale.
                if (instantiatedTreeCount < largeCount) {
                    hex = (GameObject) Instantiate(LargeTreeHexes[Random.Range(0, LargeTreeHexes.Length)], pos, rot);
                    instantiatedTreeCount++;
                } else if (instantiatedTreeCount < largeCount + mediumCount) {
                    hex = (GameObject) Instantiate(MediumTreeHexes[Random.Range(0, MediumTreeHexes.Length)], pos, rot);
                    instantiatedTreeCount++;
                } else if (instantiatedTreeCount < largeCount + mediumCount + smallCount) {
                    hex = (GameObject) Instantiate(SmallTreeHexes[Random.Range(0, SmallTreeHexes.Length)], pos, rot);
                    instantiatedTreeCount++;
                } else {
                    hex = (GameObject) Instantiate(LargeTreeHexes[Random.Range(0, LargeTreeHexes.Length)], pos, rot);
                    // back to large tree and reset treecount
                    instantiatedTreeCount = 1;
                }
                hex.transform.parent = hexGridGO.transform;
            }
        }
    }

    /*
    // returns hex with randomly placed trees. use this for creating new hextile prefab.takes too long in runtime creation.
    GameObject GetTreeHex() {
        GameObject hex = (GameObject) Instantiate(Hex);
        Mesh mesh = hex.GetComponent<MeshFilter>().mesh;
        Vector3[] verticies = mesh.vertices;
        for (int i = 0; i < verticies.Length; i++) {
            verticies[i] = hex.transform.TransformPoint(verticies[i]);
        }
        int treeCount = Random.Range(5, 10);
        int numTreeSet = 0;
        while (numTreeSet < treeCount) {
            int chosenVerticeIndex = Random.Range(0, verticies.Length);
            int nextVertice;
            if (chosenVerticeIndex == verticies.Length - 1) {
                nextVertice = 0;
            } else {
                nextVertice = chosenVerticeIndex + 1;
            }
            Vector3 randPointBtwVerticies = Vector3.Lerp(verticies[chosenVerticeIndex], verticies[nextVertice], Random.Range(0f, 1f));
            //Vector3 randPointBtwVerticies = verticies[chosenVerticeIndex] + (verticies[nextVertice] - verticies[chosenVerticeIndex]) * Random.Range(0f, 1f);
            Vector3 randPointBtwVerticeAndCenter = Vector3.Lerp(randPointBtwVerticies, hex.transform.position, Random.Range(0f, 1f));
            //Vector3 randPointBtwVerticeAndCenter = randPointBtwVerticies + (hex.transform.position - randPointBtwVerticies) * Random.Range(0f, 1f);

            //hex.transform.TransformPoint(verticies[chosenVerticeIndex]);
            if (!Physics.CheckSphere(randPointBtwVerticeAndCenter, 1.5f)) {
                Quaternion randRot = Quaternion.Euler(0, Random.Range(0, 30), 0);
                GameObject tempTree = (GameObject)Instantiate(trees[Random.Range(0, trees.Length)], randPointBtwVerticeAndCenter, randRot);
                //tempTree.transform.rotation = randRot;
                //tempTree.transform.eulerAngles.y = Random.Range(0, 360);
                tempTree.transform.parent = hex.transform;
                numTreeSet++;
            }
        }
        return hex;
    }
    */

    void Start() {
        setSizes();
        createGrid();
        // hide hexgrid
        foreach (Transform hg in GameObject.Find("HexGrid").transform) {
            hg.GetComponent<MeshRenderer>().enabled = false;
        }
    }
}